from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from f110_msgs.msg import Wpnt
from frenet_converter.frenet_converter import FrenetConverter
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
# plot raceline, adding arrow at first waypoint for general direction
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from single_track_mpc.STMPC import STMPC


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
    return np.array(waypoints_list)


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
    Returns np.nan if a full lap wasn't completed within x_hist.
    """
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
    return float(np.mean(np.abs(x_hist[:, idx_n])))


def compute_mean_abs_speed_error(waypoints: np.ndarray,
                                 x_hist: np.ndarray,
                                 track_len: float,
                                 idx_s: int = 0,
                                 idx_vx: int = 3) -> float:
    """
    Mean absolute error between actual vx and target vx(s).
    waypoints columns expected: [x, y, vx_target, kappa, s, d_left, d_right]
    """
    s_seq = x_hist[:, idx_s] % track_len
    vx_seq = x_hist[:, idx_vx]
    wp_s = waypoints[:, 4]
    wp_v = waypoints[:, 2]
    idxs = np.abs(wp_s[None, :] - s_seq[:, None]).argmin(axis=1)
    v_targets = wp_v[idxs]
    return float(np.mean(np.abs(vx_seq - v_targets)))


def plot_mu_sensitivity_violin(delta_mu,
                               lap_times,
                               mean_abs_lat_dev,
                               mean_abs_vel_err,
                               mean_cost=None,
                               colors=("tab:blue", "tab:green", "tab:red", "tab:purple")):
    """
    Plot violin + scatter + mean line for metrics vs Δμ.
    Supports an optional 4th metric: mean_cost (objective).
    """
    def group_by_mu(arr):
        a = np.asarray(arr)
        if a.ndim == 1:
            return [np.array([v]) for v in a]
        if a.shape[1] <= a.shape[0]:
            return [a[:, j] for j in range(a.shape[1])]
        return [a[j, :] for j in range(a.shape[0])]

    lt_groups = group_by_mu(lap_times)
    lat_groups = group_by_mu(mean_abs_lat_dev)
    vel_groups = group_by_mu(mean_abs_vel_err)
    groups_all = [lt_groups, lat_groups, vel_groups]
    titles_all = ["Lap time [s]", "Mean |n| [m]", "Mean |vx−v*| [m/s]"]
    if mean_cost is not None:
        cost_groups = group_by_mu(mean_cost)
        groups_all = [cost_groups] + groups_all
        titles_all = ["Mean cost (objective)"] + titles_all

    n_mu = len(lt_groups)
    labels = [f"{dm:+.2f}" for dm in delta_mu]
    x = np.arange(1, n_mu + 1)

    fig, axes = plt.subplots(len(groups_all), 1, figsize=(8.5, 12), sharex=True)
    if len(groups_all) == 1:
        axes = [axes]

    # choose colors for however many panels we drew
    cols = list(colors)
    if len(cols) < len(groups_all):
        cols += ["C" + str(i) for i in range(len(groups_all) - len(cols))]

    for ax, groups, col, yl in zip(axes, groups_all, cols, titles_all):
        v = ax.violinplot(groups, positions=x,
                          showmeans=False, showmedians=False, showextrema=False)
        for b in v['bodies']:
            b.set_facecolor(col)
            b.set_alpha(0.4)
            b.set_edgecolor("black")
            b.set_linewidth(0.8)

        rng = np.random.default_rng(42)
        for i, g in enumerate(groups, start=1):
            g = np.asarray(g)
            if g.size == 1:
                ax.scatter([i], g, color=col, s=40, zorder=3)
            else:
                jitter = (rng.random(g.size) - 0.5) * 0.25
                ax.scatter(i + jitter, g, color=col, alpha=0.7, s=20, zorder=3)

        means = [np.mean(np.asarray(g)) for g in groups]
        ax.plot(x, means, marker='o', color=col, linewidth=2, zorder=4)

        ax.set_ylabel(yl)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[-1].set_xlabel("Δμ (track − MPC nominal)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import time

    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")
    # global_traj_wpnts_mintime.wpnts  # global_traj_wpnts_iqp.wpnts
    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_mintime.wpnts
    refline_xy = np.array([(p.x_m, p.y_m) for p in refline])
    left_dist = np.array([p.d_left for p in refline])
    right_dist = np.array([p.d_right for p in refline])

    # Car, track and mpc parameters
    racecar_version = "NUC5"
    floor = "dubi"
    init_pose_frenet = np.array([20.0, 0.0, 0.0])
    pose_frenet = init_pose_frenet
    vx_initial = 3.0
    waypoints = extract_waypoints_data(refline)
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config = load_trailing_config_ros(racecar_version)

    iterations = 500
    integration = "rk4"

    mpc = STMPC(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version="f110",
        stmpc_config=stmpc_config,
        car_config=car_config,
        tire_config=tire_config,
        trailing_config=trailing_config,
        controller_frequency=40.0,
        using_gokart=False
    )

    dt = 1.0 / stmpc_config.MPC_freq
    track_len = mpc.spline.track_length

    # Nominal μ used in the MPC model
    # mpc_mu_nominal = 1.0
    track_mu = 1.0

    # μ values to test in the SIMULATOR (track)
    mu_start = 0.6
    mu_end = 1.8
    mu_step = 0.1
    # track_mu_values = [factor for factor in np.arange(mu_start, mu_end + 1e-9, mu_step)]
    mpc_mu_values = [factor for factor in np.arange(mu_start, mu_end + 1e-9, mu_step)]

    delta_mu_list = []
    lap_times_list = []
    mean_abs_lat_dev_list = []
    mean_abs_vel_err_list = []
    mean_cost_list = []  # NEW: store mean objective per μ

    # 10-dim "plant" state used by _dynamics_of_car:
    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        vx_initial, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ], dtype=float)

    for mu_mpc in mpc_mu_values:
        print(f"\n=== Running closed-loop sim with MPC μ = {mu_mpc:.3f} (Δμ={mu_mpc - track_mu:+.3f}) ===")

        x_hist = []
        u_hist = []
        status_hist = []
        x_pred_seq_hist = []
        cost_hist = []   # NEW: objective per step

        # Set sim μ (track), keep MPC nominal μ unchanged
        mpc.set_sim_mu(track_mu)

        start_time = time.time()
        for k in range(iterations):
            # 1) Pin solver's initial state to current plant (first states only)
            x0_solver = x_plant[:mpc.model.n_x].copy()
            mpc.acados_solver.set(0, "lbx", x0_solver)
            mpc.acados_solver.set(0, "ubx", x0_solver)

            # Online params along horizon
            for i in range(mpc.stmpc_config.N + 1):
                idx = np.abs(waypoints[:, 4] - (mpc.mpc_sd[i, 0] % track_len)).argmin()
                target_v_speed = waypoints[idx, 2]
                online_parameters = np.array([
                    target_v_speed,
                    stmpc_config.qadv,
                    stmpc_config.qv,
                    stmpc_config.qn,
                    stmpc_config.qalpha,
                    stmpc_config.qjerk,
                    stmpc_config.qddelta,
                    stmpc_config.alat_max,
                    stmpc_config.a_min,
                    stmpc_config.a_max,
                    stmpc_config.track_safety_margin,
                    0,
                    mu_mpc,
                    stmpc_config.terminal_multiplier
                ], dtype=np.float64)
                mpc.acados_solver.set(i, "p", online_parameters)

            # 2) Solve MPC
            status = mpc.acados_solver.solve()
            status_hist.append(status)

            # NEW: read objective value (total cost) for this iteration
            try:
                obj_val = float(mpc.acados_solver.get_cost())
            except Exception:
                # Fallback in case API differs; set NaN so it won't break averaging
                obj_val = np.nan
            cost_hist.append(obj_val)

            if not (status == 0 or status == 2):
                mpc.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
                print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

            # 3) Read optimal control (stage 0): u = [jerk, ddelta]
            u0 = mpc.acados_solver.get(0, "u").astype(float)
            u_hist.append(u0.copy())

            # Collect full horizon state predictions
            H = mpc.stmpc_config.N
            nx = mpc.model.n_x
            x_pred_full = np.zeros((H + 1, nx))
            for i in range(H + 1):
                x_pred_full[i, :] = mpc.acados_solver.get(i, "x")
            x_pred_seq_hist.append(x_pred_full)

            mpc.mpc_sd = np.array([
                mpc.acados_solver.get(j, "x")[:2] for j in range(mpc.stmpc_config.N + 1)
            ])

            # 4) Apply control to the "plant":
            x_plant[8] = u0[0]  # jerk
            x_plant[9] = u0[1]  # derDelta

            if integration == "rk4":
                x_plant = rk4_step(mpc, x_plant, dt)
            else:
                xdot = mpc._dynamics_of_car(0.0, x_plant)
                x_plant = x_plant + dt * np.array(xdot, dtype=float)

            # 5) Log plant state
            x_hist.append(x_plant.copy())

            elapsed_time = time.time() - start_time
            start_time = time.time()

        x_hist = np.asarray(x_hist)
        u_hist = np.asarray(u_hist)
        x_pred_seq_hist = np.asarray(x_pred_seq_hist)
        cost_hist = np.asarray(cost_hist, dtype=float)

        # reset plant for next μ
        x_plant = np.array([
            init_pose_frenet[0], init_pose_frenet[1], init_pose_frenet[2],
            vx_initial, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ], dtype=float)

        # Reset MPC internals
        mpc.mpc_sd = np.zeros((mpc.stmpc_config.N + 1, 2))
        mpc.mpc_init_params()
        mpc.apply_warm_start(pose_frenet=init_pose_frenet)
        mpc.acados_solver.set(0, "x", x_plant[:nx])

        # ---- Metrics for this μ ----
        lt = compute_lap_time(x_hist, dt, track_len, idx_s=0)
        mad_n = compute_mean_abs_lateral_dev(x_hist, idx_n=1)
        mae_v = compute_mean_abs_speed_error(waypoints, x_hist, track_len, idx_s=0, idx_vx=3)
        mean_cost = float(np.nanmean(cost_hist[np.isfinite(cost_hist)])) if cost_hist.size else np.nan  # NEW

        delta_mu_list.append(mu_mpc - track_mu)
        lap_times_list.append(lt)
        mean_abs_lat_dev_list.append(mad_n)
        mean_abs_vel_err_list.append(mae_v)
        mean_cost_list.append(mean_cost)  # NEW

    # Arrays
    delta_mu = np.asarray(delta_mu_list, dtype=float)
    lap_times = np.asarray(lap_times_list, dtype=float)
    mean_abs_lat_dev = np.asarray(mean_abs_lat_dev_list, dtype=float)
    mean_abs_vel_err = np.asarray(mean_abs_vel_err_list, dtype=float)
    mean_cost_arr = np.asarray(mean_cost_list, dtype=float)  # NEW

    # Sort by Δμ
    order = np.argsort(delta_mu)
    delta_mu = delta_mu[order]
    lap_times = lap_times[order]
    mean_abs_lat_dev = mean_abs_lat_dev[order]
    mean_abs_vel_err = mean_abs_vel_err[order]
    mean_cost_arr = mean_cost_arr[order]  # NEW

    # Plot sensitivity (now with mean cost)
    colors = ("royalblue", "seagreen", "crimson", "purple")
    fig = plot_mu_sensitivity_violin(
        delta_mu,
        lap_times,
        mean_abs_lat_dev,
        mean_abs_vel_err,
        mean_cost=mean_cost_arr,   # NEW
        colors=colors
    )
    plt.show()
