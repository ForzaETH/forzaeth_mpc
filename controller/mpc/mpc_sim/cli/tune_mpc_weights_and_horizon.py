#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOT WORKING YET!!!!!!!

OPTUNA AUTOTUNING — MPC cost weights + horizon length (H)

Same behavior as your previous script, but now H is a tuned parameter.
When H changes, we (re)build the ACADOS solver/stack for that horizon.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from tqdm.auto import tqdm

from ..core.controllers import ControllerMPC
from ..core.plant import Plant
from ..core.rollout import RolloutRunner
from ..core.scenario import Scenario
# Metrics & viz (unchanged)
from ..metrics.metrics import (lap_time, mean_abs_speed_error,
                               mean_abs_track_error)
from ..metrics.metrics import mean_cost as metric_mean_cost
from ..plotting.vis_tune_mpc_weights import plot_parallel_params_vs_metrics

# ----------------------------
# Search space (Optuna params)
# ----------------------------
PARAM_SPECS: Dict[str, Dict] = {
    "q_adv": dict(low=1.0, high=10.0, log=True),
    "q_v": dict(low=1e-1, high=10.0, log=True),
    "q_n": dict(low=5.0, high=50.0, log=True),
    "q_alpha": dict(low=1e-6, high=1.0, log=True),
    "q_jerk": dict(low=1e-6, high=1.0, log=True),
    "q_ddelta": dict(low=1e-6, high=1.0, log=True),
    "terminal_multiplier": dict(low=1e-2, high=10.0, log=True),

    # NEW: Horizon length H (integer). Adjust bounds if you want.
    "H": {"low": 20, "high": 80, "is_int": True},
}
PARAM_ORDER = list(PARAM_SPECS.keys())


def suggest_params(trial: optuna.Trial) -> Tuple[Dict[str, float], int]:
    """
    Suggest weights (qdict) and horizon length H.
    We keep H out of qdict to avoid pushing it into the p-vector.
    """
    qdict: Dict[str, float] = {}
    H: Optional[int] = None

    for name, spec in PARAM_SPECS.items():
        if spec.get("is_int", False):
            val = trial.suggest_int(name, spec["low"], spec["high"])
            if name == "H":
                H = int(val)
        else:
            qdict[name] = float(
                trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            )
    assert H is not None, "Internal error: H must be suggested."
    return qdict, H

# ----------------------------
# Objective wrapper
# ----------------------------


class MPCOptunaObjective:
    def __init__(
        self,
        scn: Scenario,
        dt: float,
        steps: int,
        integrator: str,
        mu_track: float,
        mu_mpc: float,
        w_lt: float,
        w_cost: float,
        w_abs_n: float,
        w_abs_v: float,
        seed: int = 42,
    ):
        # Scenario + constants (unchanged from your script)
        self.scn = scn
        self.dt = float(dt)
        self.steps = int(steps)
        self.integrator = integrator
        self.mu_track = float(mu_track)
        self.mu_mpc = float(mu_mpc)
        self.w_lt = float(w_lt)
        self.w_cost = float(w_cost)
        self.w_abs_n = float(w_abs_n)
        self.w_abs_v = float(w_abs_v)

        # Cache track stuff used across trials
        self.track_len = scn.track_len
        self.waypoints = scn.waypoints

        # Will be built per-H via _build_stack
        self.mpc = None
        self.ctrl: Optional[ControllerMPC] = None
        self.plant: Optional[Plant] = None
        self.runner: Optional[RolloutRunner] = None

        # RNG (if needed)
        np.random.default_rng(seed)

    # ---------- build/rebuild stack for a given H ----------
    def _build_stack(self, H: int) -> None:
        """
        (Re)create ACADOS solver and wrappers for horizon H.
        Assumes Scenario.make_mpc accepts N=H (most setups do).
        """
        s0, n0, psi0 = [float(x) for x in self.scn.init_pose_frenet[:3]]

        # IMPORTANT: pass N=H so solver is generated for that horizon
        self.mpc = self.scn.make_mpc(
            s0=s0,
            n0=n0,
            psi0=psi0,
            dt=self.dt,
            N=int(H),   # <— key line
        )

        pose = self.scn.init_pose_frenet

        x0_full = self.scn.make_x0_full(
            self.mpc, s0=pose[0], n0=pose[1], psi0=pose[2], vx0=float(self.scn.init_vx0)
        )

        self.ctrl = ControllerMPC(self.mpc, self.track_len)     # reads H from mpc.stmpc_config.N
        self.plant = Plant(self.mpc._dynamics_of_car, dt=self.dt, integrator=self.integrator)
        self.runner = RolloutRunner(self.ctrl, self.plant, self.track_len)

        # Decouple PLANT μ
        try:
            self.mpc.set_sim_mu(self.mu_track)
        except Exception:
            pass

        return x0_full

    def _reset_mpc_state(self) -> np.ndarray:
        """
        Reset MPC/plant to scenario initial state between trials.
        Matches your previous logic.
        """
        assert self.mpc is not None and self.ctrl is not None

        pose = self.scn.init_pose_frenet

        x0_full = self.scn.make_x0_full(
            self.mpc, s0=pose[0], n0=pose[1], psi0=pose[2], vx0=float(self.scn.init_vx0)
        )

        try:
            self.mpc.acados_solver.set(0, "x", x0_full[: self.mpc.model.n_x])
        except Exception:
            pass

        return x0_full

    def __call__(self, trial: optuna.Trial) -> float:
        # 1) Suggest hyperparams
        qdict, H = suggest_params(trial)

        # 2) Build or rebuild stack if H changed
        if (self.ctrl is None) or (self.ctrl.H != int(H)):
            tqdm.write(f"[rebuild] Using horizon H={int(H)}")
            x0_full = self._build_stack(int(H))

        # 3) Reset state
        # x0_full = self._reset_mpc_state()

        # 4) Rollout (unchanged paths/signatures)
        X, U, C, S, _ = self.runner.run(
            x0_full=x0_full,
            waypoints=self.waypoints,
            iterations=self.steps,
            qdict=qdict,
            mpc_mu=self.mu_mpc,
            set_sim_mu_fn=None,
            capture_predictions=False,
            log_fn=tqdm.write,
        )

        # 5) Metrics (same as before)
        lt = lap_time(X, self.dt, self.track_len, idx_s=0)
        mc = metric_mean_cost(C)
        man = mean_abs_track_error(X, idx_n=1)
        mve = mean_abs_speed_error(self.waypoints, X, self.track_len, idx_s=0, idx_vx=3)

        # Penalties for invalid runs
        lt = 1e6 if math.isnan(lt) else float(lt)
        mc = 1e6 if math.isnan(mc) else float(mc)
        man = 1e6 if math.isnan(man) else float(man)
        mve = 1e6 if math.isnan(mve) else float(mve)

        # 6) Scalarized objective (unchanged)
        J = self.w_lt * lt + self.w_cost * mc + self.w_abs_n * man + self.w_abs_v * mve

        # 7) Log metrics
        trial.set_user_attr("H", int(H))
        trial.set_user_attr("lap_time", lt)
        trial.set_user_attr("mean_cost", mc)
        trial.set_user_attr("mean_abs_n", man)
        trial.set_user_attr("mean_abs_v_err", mve)

        # Optional pruning on lap time (kept as your original style)
        trial.report(lt, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(J)

# ----------------------------
# CLI (unchanged)
# ----------------------------


def main():
    p = argparse.ArgumentParser(description="Optuna autotuning for MPC cost weights (now also tunes H).")
    # Scenario / build
    p.add_argument("--track", type=str, default="THETRACK")
    p.add_argument("--racecar_version", type=str, default="NUC5")
    p.add_argument("--floor", type=str, default="dubi")
    p.add_argument("--refline", type=str, default="iqp", choices=["iqp", "mintime"])
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "euler"])

    # Initial pose
    p.add_argument("--s0", type=float, default=20.0)
    p.add_argument("--n0", type=float, default=0.0)
    p.add_argument("--psi0", type=float, default=0.0)
    p.add_argument("--vx0", type=float, default=3.0)

    # Friction split
    p.add_argument("--mu_track", type=float, default=1.00, help="Plant/track μ via set_sim_mu")
    p.add_argument("--mu_mpc", type=float, default=1.00, help="MPC μ used in p-vector")

    # Scalarization weights
    p.add_argument("--w_lt", type=float, default=1.0, help="Weight for lap time in objective")
    p.add_argument("--w_cost", type=float, default=0.0, help="Weight for mean solver cost in objective")
    p.add_argument("--w_abs_n", type=float, default=0.0, help="Weight for mean solver lateral error in objective")
    p.add_argument("--w_abs_v", type=float, default=0.0, help="Weight for mean solver velocity error in objective")

    # Optuna
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--study_name", type=str, default="mpc_cost_autotune")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///study.db")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pruner", action="store_true", help="Enable MedianPruner")

    # visualization
    p.add_argument(
        "--no_vis_optimization_history",
        dest="vis_optimization_history",
        action="store_false",
        help="Disable visualization of the optimization history."
    )
    p.add_argument("--vis_parallel", action="store_true")
    args = p.parse_args()

    # Build scenario with your initial pose baked in (unchanged)
    scn = Scenario(track=args.track,
                   racecar_version=args.racecar_version,
                   floor=args.floor, refline=args.refline)
    scn.init_pose_frenet = np.array([args.s0, args.n0, args.psi0], dtype=float)
    scn.init_vx0 = float(args.vx0)

    objective = MPCOptunaObjective(
        scn=scn,
        dt=args.dt,
        steps=args.steps,
        integrator=args.integrator,
        mu_track=args.mu_track,
        mu_mpc=args.mu_mpc,
        w_lt=args.w_lt,
        w_cost=args.w_cost,
        w_abs_n=args.w_abs_n,
        w_abs_v=args.w_abs_v,
        seed=args.seed,
    )

    # Create study (unchanged)
    pruner = optuna.pruners.MedianPruner() if args.pruner else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            pruner=pruner,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            pruner=pruner,
            sampler=sampler,
        )

    print("=== Starting optimization ===")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n=== Best Trial ===")
    bt = study.best_trial
    print(f"Value (J): {bt.value:.6f}")
    print("Weights + H:")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")
    print("Metrics:")
    print(f"  lap_time:        {bt.user_attrs.get('lap_time')}")
    print(f"  mean_cost:       {bt.user_attrs.get('mean_cost')}")
    print(f"  mean_abs_n:      {bt.user_attrs.get('mean_abs_n')}")
    print(f"  mean_abs_v_err:  {bt.user_attrs.get('mean_abs_v_err')}")

    # Optimization history (unchanged)
    if args.vis_optimization_history:
        try:
            from optuna.visualization.matplotlib import \
                plot_optimization_history
            ax = plot_optimization_history(study)
            ax.set_title("Optuna Optimization History")
            ax.set_xlabel("trial")
            ax.set_ylabel("objective J")
            ax.set_ylim(0.1, 10)
            fig = ax.get_figure()
            fig.tight_layout()
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    # Parallel coord plot (unchanged; now includes H because it’s in PARAM_ORDER)
    if args.vis_parallel:
        param_names = PARAM_ORDER
        param_limits = {k: (spec["low"], spec["high"]) for k, spec in PARAM_SPECS.items()}
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if len(trials) >= 2:
            P = np.array([[t.params.get(name, np.nan) for name in param_names] for t in trials], dtype=float)
            lt = np.array([t.user_attrs.get("lap_time", np.nan) for t in trials], dtype=float)
            man = np.array([t.user_attrs.get("mean_abs_n", np.nan) for t in trials], dtype=float)
            mc = np.array([t.user_attrs.get("mean_cost", np.nan) for t in trials], dtype=float)

            lt_cap = 1.5 * np.nanmedian(lt) if np.isfinite(np.nanmedian(lt)) else np.nanmax(lt)
            mc_cap = 1.5 * np.nanmedian(mc) if np.isfinite(np.nanmedian(mc)) else np.nanmax(mc)
            man_cap = 1.5 * np.nanmedian(man) if np.isfinite(np.nanmedian(man)) else np.nanmax(man)
            lt = np.clip(lt, 0, lt_cap)
            mc = np.clip(mc, 0, mc_cap)
            man = np.clip(man, 0, man_cap)

            metrics_matrix = np.column_stack([lt, man, mc])
            metric_names = ["Lap time [s]", "Mean |n| [m]", "Cost"]

            ok = np.isfinite(P).all(axis=1) & np.isfinite(metrics_matrix).all(axis=1)
            P = P[ok]
            metrics_matrix = metrics_matrix[ok]
            mc = mc[ok]

            fig = plot_parallel_params_vs_metrics(
                params_matrix=P,
                param_names=param_names,
                metrics_matrix=metrics_matrix,
                metric_names=metric_names,
                color_by=mc,
                cmap="viridis",
                alpha=0.5,
                lw=1.0,
                fig_size=(14, 4),
                n_y_ticks=5,
                title="Tunable parameters vs performance (color = Cost)",
                fixed_param_limits=param_limits,
            )
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("[parallel-plot] Not enough completed trials to plot (need >= 2).")


if __name__ == "__main__":
    main()
