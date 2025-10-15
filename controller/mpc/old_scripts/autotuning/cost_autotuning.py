# optuna_tune_mpc.py
# Autotunes MPC cost weights using Optuna to minimize: J = w_lt * lap_time + w_cost * mean_cost
# One rollout per trial: set weights -> simulate -> measure -> report -> suggest new weights

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple

# (Optional) comment out if you don't want plots during trials
import matplotlib.pyplot as plt
import numpy as np
import optuna
from f110_msgs.msg import Wpnt
from frenet_converter.frenet_converter import FrenetConverter
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from single_track_mpc.STMPC import STMPC

# -----------------------
# Helpers (copied/adapted)
# -----------------------


def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    """Convert WpntArray to numpy array for easier processing."""
    waypoints_list = [
        (wpnt.x_m,
         wpnt.y_m,
         wpnt.vx_mps,
         wpnt.kappa_radpm,
         wpnt.s_m,
         wpnt.d_left,
         wpnt.d_right) for wpnt in waypoints]
    return np.array(waypoints_list, dtype=float)


def rk4_step(mpc, x: np.ndarray, dt: float) -> np.ndarray:
    """
    One Runge-Kutta 4 step for the plant dynamics.

    mpc._dynamics_of_car(t, x) must return dx/dt for a given state x (10-dim).
    """
    def f(xx): return np.array(mpc._dynamics_of_car(0.0, xx), dtype=float)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _unwrap_s_progress(s_seq: np.ndarray, track_len: float) -> np.ndarray:
    """Unwrap s (mod track_len) into a monotonic cumulative progress."""
    s_unwrapped = np.zeros_like(s_seq, dtype=float)
    s_unwrapped[0] = s_seq[0]
    for k in range(1, len(s_seq)):
        ds = s_seq[k] - s_seq[k - 1]
        if ds < -0.5 * track_len:
            ds += track_len
        elif ds > 0.5 * track_len:
            ds -= track_len
        s_unwrapped[k] = s_unwrapped[k - 1] + ds
    return s_unwrapped


def compute_lap_time(x_hist: np.ndarray, dt: float, track_len: float, idx_s: int = 0) -> float:
    """
    Compute time to complete one lap since start index 0.
    Returns np.nan if a full lap wasn't completed.
    """
    if x_hist.size == 0:
        return float('nan')
    s_raw = x_hist[:, idx_s]
    s_unwrapped = _unwrap_s_progress(s_raw, track_len)
    s0 = s_unwrapped[0]
    target = s0 + track_len
    cross_idx = np.argmax(s_unwrapped >= target)
    if s_unwrapped[cross_idx] < target:
        return float('nan')
    return cross_idx * dt


def compute_mean_abs_lateral_dev(x_hist: np.ndarray, idx_n: int = 1) -> float:
    """Mean absolute lateral deviation (|n|)."""
    return float(np.mean(np.abs(x_hist[:, idx_n]))) if x_hist.size else float('nan')


def compute_mean_abs_speed_error(waypoints: np.ndarray,
                                 x_hist: np.ndarray,
                                 track_len: float,
                                 idx_s: int = 0,
                                 idx_vx: int = 3) -> float:
    """
    Mean absolute error between actual vx and target vx(s).
    waypoints columns: [x, y, vx_target, kappa, s, d_left, d_right]
    """
    if x_hist.size == 0:
        return float('nan')
    s_seq = x_hist[:, idx_s] % track_len
    vx_seq = x_hist[:, idx_vx]
    wp_s = waypoints[:, 4]
    wp_v = waypoints[:, 2]
    idxs = np.abs(wp_s[None, :] - s_seq[:, None]).argmin(axis=1)
    v_targets = wp_v[idxs]
    return float(np.mean(np.abs(vx_seq - v_targets)))


# -----------------------
# Cost-weights injection
# -----------------------

def build_weight_space(trial: optuna.Trial) -> Dict[str, float]:
    """
    Define/search your MPC weighting hyperparameters here.
    You can rename/add/remove as you see fit.
    """
    weights = {
        # state penalties
        "q_adv": trial.suggest_float("q_adv", 1e-6, 10.0, log=True),        # advance/progress weighting
        "q_v": trial.suggest_float("q_v", 5.0, 40.0, log=True),           # speed tracking
        "q_n": trial.suggest_float("q_n", 5.0, 50.0, log=True),           # lateral deviation
        "q_alpha": trial.suggest_float("q_alpha", 1e-6, 1.0, log=True),    # heading/curvature style
        # input/slew penalties
        "q_jerk": trial.suggest_float("q_jerk", 1e-6, 1.0, log=True),
        "q_ddelta": trial.suggest_float("q_ddelta", 1e-6, 1.0, log=True),
    }
    return weights

# -----------------------
# Single rollout simulator
# -----------------------


def simulate_one_rollout(
    mpc: STMPC,
    waypoints: np.ndarray,
    track_len: float,
    dt: float,
    iterations: int,
    track_mu: float,
    mpc_mu: float,
    integrator: str = "rk4",
    trial_weights: dict = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Runs one closed-loop simulation with the current MPC settings.
    Returns (lap_time, mean_cost, extra_metrics).
    mean_cost is the average of acados solver's objective per step (ignoring NaNs).
    """
    # initial plant state (10-dim expected by your _dynamics_of_car)
    #   state layout assumed same as your baseline
    x_plant = np.array([
        mpc.pose_frenet[0], mpc.pose_frenet[1], mpc.pose_frenet[2],
        mpc.vx_initial, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ], dtype=float)

    x_hist = []
    cost_hist = []

    # configure μ
    mpc.set_sim_mu(track_mu)

    # NOTE: solver pin/online params inside the loop (as in your baseline)
    H = mpc.stmpc_config.N
    nx = mpc.model.n_x

    for k in range(iterations):
        # 1) Pin initial state
        x0_solver = x_plant[:nx].copy()
        mpc.acados_solver.set(0, "lbx", x0_solver)
        mpc.acados_solver.set(0, "ubx", x0_solver)

        # 2) Online parameters along horizon (keeping your baseline structure)
        for i in range(H + 1):
            idx = np.abs(waypoints[:, 4] - (mpc.mpc_sd[i, 0] % track_len)).argmin()
            target_v_speed = waypoints[idx, 2]
            # If trial_weights provided, use them; else fall back to config
            q_adv = trial_weights.get("q_adv", mpc.stmpc_config.qadv) if trial_weights else mpc.stmpc_config.qadv
            q_v = trial_weights.get("q_v", mpc.stmpc_config.qv) if trial_weights else mpc.stmpc_config.qv
            q_n = trial_weights.get("q_n", mpc.stmpc_config.qn) if trial_weights else mpc.stmpc_config.qn
            q_alpha = trial_weights.get(
                "q_alpha", mpc.stmpc_config.qalpha) if trial_weights else mpc.stmpc_config.qalpha
            q_jerk = trial_weights.get("q_jerk", mpc.stmpc_config.qjerk) if trial_weights else mpc.stmpc_config.qjerk
            q_ddelta = trial_weights.get("q_ddelta",
                                         mpc.stmpc_config.qddelta) if trial_weights else mpc.stmpc_config.qddelta

            online_parameters = np.array([
                target_v_speed,
                q_adv,
                q_v,
                q_n,
                q_alpha,
                q_jerk,
                q_ddelta,
                mpc.stmpc_config.alat_max,
                mpc.stmpc_config.a_min,
                mpc.stmpc_config.a_max,
                mpc.stmpc_config.track_safety_margin,
                0,
                mpc_mu
            ], dtype=np.float64)

            mpc.acados_solver.set(i, "p", online_parameters)

        # 3) Solve
        status = mpc.acados_solver.solve()
        if not (status == 0 or status == 2):
            mpc.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])

        # 4) Read objective value
        try:
            obj_val = float(mpc.acados_solver.get_cost())
        except Exception:
            obj_val = np.nan
        cost_hist.append(obj_val)

        # 5) Apply u0 to plant
        u0 = mpc.acados_solver.get(0, "u").astype(float)
        x_plant[8] = u0[0]  # jerk
        x_plant[9] = u0[1]  # derDelta

        if integrator == "rk4":
            x_plant = rk4_step(mpc, x_plant, dt)
        else:
            xdot = mpc._dynamics_of_car(0.0, x_plant)
            x_plant = x_plant + dt * np.array(xdot, dtype=float)

        x_hist.append(x_plant.copy())

        # update stored horizon (your usual mechanism)
        mpc.mpc_sd = np.array([
            mpc.acados_solver.get(j, "x")[:2] for j in range(H + 1)
        ])

        # simple early exit if lap already completed
        if len(x_hist) > 5:
            lt_tmp = compute_lap_time(np.asarray(x_hist), dt, track_len, idx_s=0)
            if not math.isnan(lt_tmp):
                break

    x_hist = np.asarray(x_hist)
    cost_hist = np.asarray(cost_hist, dtype=float)

    # metrics
    lap_time = compute_lap_time(x_hist, dt, track_len, idx_s=0)
    mean_cost = float(np.nanmean(cost_hist[np.isfinite(cost_hist)])) if cost_hist.size else float('nan')
    mad_n = compute_mean_abs_lateral_dev(x_hist, idx_n=1)
    mae_v = compute_mean_abs_speed_error(waypoints, x_hist, track_len, idx_s=0, idx_vx=3)

    return lap_time, mean_cost, {"mean_abs_n": mad_n, "mean_abs_v_err": mae_v}


# -----------------------
# Optuna Objective
# -----------------------

class MPCStudyObjective:
    def __init__(
        self,
        track_name: str,
        racecar_version: str,
        floor: str,
        init_pose_frenet: Tuple[float, float, float],
        vx_initial: float,
        mpc_mu: float,
        track_mu: float,
        iterations: int,
        controller_frequency: float,
        w_lt: float,
        w_cost: float,
        integrator: str = "rk4",
        using_gokart: bool = False,
        seed: int = 42
    ):
        # Load track
        gb_waypoints: GlobalWaypoints = read_global_waypoints(track_name)
        # global_traj_wpnts_mintime.wpnts  # global_traj_wpnts_iqp.wpnts
        refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_mintime.wpnts
        self.waypoints = extract_waypoints_data(refline)
        self.refline = refline

        # Configs
        self.stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
        self.car_config: CarConfig = load_car_config_ros(racecar_version)
        self.tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
        self.trailing_config = load_trailing_config_ros(racecar_version)

        # Build MPC instance
        self.mpc = STMPC(
            pose_frenet=np.array(init_pose_frenet, dtype=float),
            waypoints_data=self.waypoints,
            racecar_version="f110",
            stmpc_config=self.stmpc_config,
            car_config=self.car_config,
            tire_config=self.tire_config,
            trailing_config=self.trailing_config,
            controller_frequency=controller_frequency,
            using_gokart=using_gokart
        )
        self.mpc.pose_frenet = np.array(init_pose_frenet, dtype=float)
        self.mpc.vx_initial = vx_initial

        # Timing / track
        self.dt = 1.0 / self.stmpc_config.MPC_freq
        self.track_len = self.mpc.spline.track_length

        # Fixed scenario knobs
        self.iterations = iterations
        self.mpc_mu = mpc_mu
        self.track_mu = track_mu
        self.integrator = integrator

        # Scalarization weights
        self.w_lt = w_lt
        self.w_cost = w_cost

        # Seed for any randomness (if any used)
        np.random.default_rng(seed)

    def __call__(self, trial: optuna.Trial) -> float:
        # 1) Propose weights
        weights = build_weight_space(trial)

        # 3) Reset MPC internals before rollout (fresh start)
        #    Reinitialize horizon state guesses etc.
        self.mpc.mpc_sd = np.zeros((self.mpc.stmpc_config.N + 1, 2))
        self.mpc.mpc_init_params()
        self.mpc.apply_warm_start(pose_frenet=self.mpc.pose_frenet)
        nx = self.mpc.model.n_x
        x0_full = np.array([
            self.mpc.pose_frenet[0], self.mpc.pose_frenet[1], self.mpc.pose_frenet[2],
            self.mpc.vx_initial, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ], dtype=float)
        self.mpc.acados_solver.set(0, "x", x0_full[:nx])

        # 4) Rollout
        lt, mean_cost, extra = simulate_one_rollout(
            mpc=self.mpc,
            waypoints=self.waypoints,
            track_len=self.track_len,
            dt=self.dt,
            iterations=self.iterations,
            track_mu=self.track_mu,
            mpc_mu=self.mpc_mu,
            integrator=self.integrator,
            trial_weights=weights,
        )

        # 5) Handle failures with large penalties (also allows pruning)
        if math.isnan(lt):
            # did not complete a lap → heavy penalty
            lt = 1e6
        if math.isnan(mean_cost):
            mean_cost = 1e6

        # 6) Scalarized objective
        J = self.w_lt * lt  # self.w_cost * mean_cost or self.w_lt * lt

        # 7) Report intermediate values for multi-objective visibility / diagnostics
        trial.set_user_attr("lap_time", float(lt))
        trial.set_user_attr("mean_cost", float(mean_cost))
        trial.set_user_attr("mean_abs_n", float(extra["mean_abs_n"]))
        trial.set_user_attr("mean_abs_v_err", float(extra["mean_abs_v_err"]))

        # For pruning based on lap time alone, for example:
        trial.report(lt, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(J)


# -----------------------
# CLI / main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna autotuning for MPC cost weights.")
    parser.add_argument("--track", type=str, default="THETRACK")
    parser.add_argument("--racecar_version", type=str, default="NUC5")
    parser.add_argument("--floor", type=str, default="dubi_hoons")
    parser.add_argument("--init_s", type=float, default=10.0)
    parser.add_argument("--init_n", type=float, default=0.0)
    parser.add_argument("--init_alpha", type=float, default=0.0)
    parser.add_argument("--vx_initial", type=float, default=3.0)
    parser.add_argument("--mpc_mu", type=float, default=1.0, help="Nominal μ used inside the MPC model")
    parser.add_argument("--track_mu", type=float, default=1.0, help="Ground-truth μ used by the plant dynamics")
    parser.add_argument("--iterations", type=int, default=400, help="Max control steps per rollout")
    parser.add_argument("--controller_frequency", type=float, default=40.0)
    parser.add_argument("--integrator", type=str, choices=["rk4", "euler"], default="rk4")
    parser.add_argument("--trials", type=int, default=500, help="Number of Optuna trials")
    parser.add_argument("--w_lt", type=float, default=1.0, help="Weight for lap time in objective")
    parser.add_argument("--w_cost", type=float, default=0.1, help="Weight for mean solver cost in objective")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--study_name", type=str, default="mpc_cost_autotune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pruner", action="store_true", help="Enable MedianPruner")
    args = parser.parse_args()

    init_pose = (args.init_s, args.init_n, args.init_alpha)

    objective = MPCStudyObjective(
        track_name=args.track,
        racecar_version=args.racecar_version,
        floor=args.floor,
        init_pose_frenet=init_pose,
        vx_initial=args.vx_initial,
        mpc_mu=args.mpc_mu,
        track_mu=args.track_mu,
        iterations=args.iterations,
        controller_frequency=args.controller_frequency,
        w_lt=args.w_lt,
        w_cost=args.w_cost,
        integrator=args.integrator,
        using_gokart=False,
        seed=args.seed,
    )

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner(),
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner(),
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

    print("=== Starting optimization ===")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Best Trial ===")
    bt = study.best_trial
    print(f"Value (J): {bt.value:.6f}")
    print("Weights:")
    for k, v in bt.params.items():
        print(f"  {k}: {v:.6g}")
    print("Metrics:")
    print(f"  lap_time:  {bt.user_attrs.get('lap_time')}")
    print(f"  mean_cost: {bt.user_attrs.get('mean_cost')}")
    print(f"  mean_abs_n: {bt.user_attrs.get('mean_abs_n')}")
    print(f"  mean_abs_v_err: {bt.user_attrs.get('mean_abs_v_err')}")

    # (Optional) plot optimization history if matplotlib available
    # try:
    from optuna.visualization.matplotlib import plot_optimization_history
    ax = plot_optimization_history(study)
    ax.set_ylim(0, 10)               # <-- limit y-axis
    fig = ax.get_figure()
    fig.suptitle("Optuna Optimization History")
    fig.tight_layout()
    fig.savefig("optimization_history.png")
    plt.show()
    # except Exception:
    #     pass


if __name__ == "__main__":
    main()
