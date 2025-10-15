from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from f110_msgs.msg import Wpnt
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

# Initialize MPC
# =========================
# NEW: metric helper funcs
# =========================


def _unwrap_s_progress(s_seq: np.ndarray, track_len: float) -> np.ndarray:
    """Unwrap s (mod track_len) into a monotonic cumulative progress."""
    s_unwrapped = np.zeros_like(s_seq, dtype=float)
    s_unwrapped[0] = s_seq[0]
    for k in range(1, len(s_seq)):
        ds = s_seq[k] - s_seq[k - 1]
        # choose the increment closest to the previous (handle wrap-around)
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
    # first index where we crossed one full lap
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
    # nearest neighbor in s for each sample
    wp_s = waypoints[:, 4]
    wp_v = waypoints[:, 2]
    idxs = np.abs(wp_s[None, :] - s_seq[:, None]).argmin(axis=1)
    v_targets = wp_v[idxs]
    return float(np.mean(np.abs(vx_seq - v_targets)))

# ======================================
# NEW: plotting function for sensitivity
# ======================================


def plot_state_error_boxplot_all(abs_errs_dict, state_labels=None,
                                 title="Model mismatch per state (one lap)",
                                 ylabel="Absolute error (units of each state)"):
    """
    abs_errs_dict: dict[mu_value] -> (M, nx) arrays (may have different M).
    Plots side-by-side boxes per state, one color per μ.
    """
    if not abs_errs_dict:
        print("[warn] No mismatch data to plot.")
        return None

    # drop empties
    abs_errs_dict = {mu: E for mu, E in abs_errs_dict.items() if E is not None and E.size > 0}
    if not abs_errs_dict:
        print("[warn] All mismatch arrays are empty.")
        return None

    mu_values = sorted(abs_errs_dict.keys())
    nx = next(iter(abs_errs_dict.values())).shape[1]
    if state_labels is None or len(state_labels) != nx:
        state_labels = [f"x{j}" for j in range(nx)]

    n_mu = len(mu_values)
    # adaptive width: keep total group width <= 0.8
    width = min(0.8 / max(n_mu, 1), 0.22)

    cmap = cm.get_cmap("tab10", n_mu)
    colors = [cmap(i) for i in range(n_mu)]

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * nx), 6))
    centers = np.arange(nx)  # tick centers per state

    for i, mu in enumerate(mu_values):
        E = abs_errs_dict[mu]  # (M, nx)
        data = [E[:, j] for j in range(nx)]
        # center groups around each state position
        offset = (i - (n_mu - 1) / 2.0) * width
        pos = centers + offset

        bp = ax.boxplot(data,
                        positions=pos,
                        widths=width * 0.9,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False)

        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color(colors[i])
                line.set_linewidth(1.2)

    ax.set_xticks(centers)
    ax.set_xticklabels(state_labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)     # <-- force range [0, 1]
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # build legend handles
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(n_mu)]
    labels = [f"μ={mu:.2f}" for mu in mu_values]
    ax.legend(handles, labels, title="MPC parameter μ", loc="upper right", frameon=True)

    ax.margins(x=0.05)
    fig.tight_layout()
    return fig


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def unwrap_pred_to_global(pred_s_raw: np.ndarray, s_ref_unwrapped: float, track_len: float) -> np.ndarray:
    """Unwrap predicted s (mod L) to a monotonic axis and align its origin close to s_ref_unwrapped."""
    s_unwrapped = np.zeros_like(pred_s_raw, dtype=float)
    s_unwrapped[0] = pred_s_raw[0]
    for i in range(1, len(pred_s_raw)):
        ds = pred_s_raw[i] - pred_s_raw[i - 1]
        if ds < -0.5 * track_len:
            ds += track_len
        elif ds > 0.5 * track_len:
            ds -= track_len
        s_unwrapped[i] = s_unwrapped[i - 1] + ds
    # align start to the global frame near s_ref_unwrapped
    k = round((s_ref_unwrapped - s_unwrapped[0]) / track_len)
    s_unwrapped += k * track_len
    return s_unwrapped


def compute_one_lap_end_index(s_meas_unwrapped: np.ndarray, track_len: float) -> int:
    """Return last index included in one full lap from the start index 0 (inclusive)."""
    start = s_meas_unwrapped[0]
    target = start + track_len
    crossed = np.where(s_meas_unwrapped >= target)[0]
    if crossed.size == 0:
        return len(s_meas_unwrapped) - 1
    return int(crossed[0])  # first index reaching ≥ one lap


def interpolate_states_along_s(s_axis: np.ndarray, X: np.ndarray, s_query: float, angle_idx=None) -> np.ndarray:
    """
    Interpolate state vector X(s) at s_query along unwrapped s_axis.
    Angle dims are unwrapped for interpolation, then rewrapped.
    """
    angle_idx = angle_idx or []
    x_interp = np.zeros(X.shape[1], dtype=float)
    # Ensure s_axis strictly increasing for np.interp (tiny epsilon if equal)
    s_axis_inc = s_axis.copy()
    for i in range(1, len(s_axis_inc)):
        if s_axis_inc[i] <= s_axis_inc[i - 1]:
            s_axis_inc[i] = s_axis_inc[i - 1] + 1e-9

    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        if j in angle_idx:
            col = np.unwrap(col)  # unwrap angle series
            val = np.interp(s_query, s_axis_inc, col)
            x_interp[j] = wrap_to_pi(val)
        else:
            x_interp[j] = np.interp(s_query, s_axis_inc, col)
    return x_interp


def compute_mismatch_abs_errors_one_lap(
    x_hist: np.ndarray,
    x_pred_seq_hist: np.ndarray,
    track_len: float,
    nx: int,
    angle_idx=None,
    s0_init: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare horizon k to measurement at the very next measurement *for that horizon*,
    which is x_hist[k]. Angle-aware, spatial interpolation.
    """
    angle_idx = angle_idx or []
    s_meas_unwrapped = _unwrap_s_progress(x_hist[:, 0], track_len)
    # first index that reaches >= one lap
    last_in_lap = compute_one_lap_end_index(s_meas_unwrapped, track_len)

    # we can only compare for k where both a horizon and x_hist[k] exist
    k_max = min(last_in_lap, len(x_pred_seq_hist) - 1, len(x_hist) - 1)
    if k_max < 0:
        return np.zeros((0, nx)), np.zeros(nx)

    abs_errs = []
    for k in range(0, k_max + 1):
        Xpred = x_pred_seq_hist[k]           # (H+1, nx), horizon starting at the pre-state for step k
        s_pred_raw = Xpred[:, 0] % track_len

        # reference s for the *start* of this horizon:
        if k == 0:
            s_ref_unwrapped = float(s0_init) if s0_init is not None else s_pred_raw[0]
        else:
            s_ref_unwrapped = s_meas_unwrapped[k - 1]

        s_pred_unw = unwrap_pred_to_global(s_pred_raw, s_ref_unwrapped, track_len)

        # measurement to compare against = "next" measurement for this horizon
        s_target = s_meas_unwrapped[k]
        x_meas = x_hist[k, :nx]

        # interpolate prediction at that target progress
        x_pred_at_target = interpolate_states_along_s(
            s_pred_unw, Xpred[:, :nx], s_target, angle_idx=angle_idx
        )

        # angle-aware error and absolute value
        err = x_meas - x_pred_at_target
        for j in angle_idx:
            err[j] = wrap_to_pi(err[j])
        abs_errs.append(np.abs(err))

    abs_errs = np.asarray(abs_errs)
    mean_abs_err = np.mean(abs_errs, axis=0) if abs_errs.size else np.zeros(nx)
    return abs_errs, mean_abs_err


def apply_normalization(abs_errs_dict, norms, clip_at_one=True):
    norms = np.where(np.asarray(norms) <= 0, 1e-9, norms)
    out = {}
    for mu, E in abs_errs_dict.items():
        En = E / norms[None, :]
        if clip_at_one:
            En = np.minimum(En, 1.0)
        out[mu] = En
    return out


def percentile_scale_from_data(abs_errs_dict_or_list, p=98, min_scale=1e-9):
    """Per-state scale from p-th percentile across ALL μ; ignores empties."""
    arrays = (list(abs_errs_dict_or_list.values())
              if isinstance(abs_errs_dict_or_list, dict) else abs_errs_dict_or_list)
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays:
        raise ValueError("No mismatch data to build percentile scales.")
    all_errs = np.concatenate(arrays, axis=0)  # (TotalM, nx)
    nx = all_errs.shape[1]
    scales = np.zeros(nx, dtype=float)
    for j in range(nx):
        scales[j] = max(np.percentile(all_errs[:, j], p), min_scale)
    return scales


def build_scale_vector_from_user(scales_by_label: dict, state_labels: list, min_scale: float = 1e-9) -> np.ndarray:
    """
    Build an (nx,) scale vector from a dict like {'v_x': 1.0, 'n': 1.0, 'theta': np.deg2rad(5), ...}.
    Any label not present uses scale=1.0 by default.
    """
    nx = len(state_labels)
    scales = np.ones(nx, dtype=float)
    for j, name in enumerate(state_labels):
        if name in scales_by_label:
            scales[j] = float(scales_by_label[name])
    scales = np.where(scales <= 0, min_scale, scales)
    return scales


def apply_normalization_manual(abs_errs_dict: dict, scales_vec: np.ndarray, clip_at_one: bool = False) -> dict:
    """
    Divide each state's absolute error by user-provided scales. Returns a new dict.
    """
    scales = np.where(np.asarray(scales_vec) <= 0, 1e-9, scales_vec)
    out = {}
    for mu, E in abs_errs_dict.items():
        if E is None or E.size == 0:
            continue
        En = E / scales[None, :]
        if clip_at_one:
            En = np.minimum(En, 1.0)
        out[mu] = En
    return out


# =========================
# Initialize MPC
# =========================

if __name__ == "__main__":
    import time

    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")
    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts
    left_dist = np.array([p.d_left for p in refline])
    right_dist = np.array([p.d_right for p in refline])

    # Car, track and mpc parameters
    racecar_version = "NUC5"
    floor = "dubi"
    init_pose_frenet = np.array([20.0, 0.0, 0.0])  # 0 position on the raceline
    pose_frenet = init_pose_frenet
    vx_initial = 3.0
    waypoints = extract_waypoints_data(refline)
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config = load_trailing_config_ros(racecar_version)

    # Simulation config parameters
    iterations = 500  # TODO: Convert it to number of laps
    integration = "rk4"  # "euler" or "rk4"

    # Build MPC (this is the model used by the solver and stays FIXED)
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

    nx = mpc.model.n_x
    state_labels = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"][:nx]
    ANGLE_IDX = [2, 5]  # theta, delta

    dt = 1.0 / stmpc_config.MPC_freq
    track_len = mpc.spline.track_length

    # -------------------------------------------------------
    # OUTER LOOP: test multiple track frictions for simulator
    # -------------------------------------------------------
    # Establish the MPC nominal mu (used by the *MPC model*, not the simulator).
    # If you store it elsewhere, set it accordingly here:
    # mpc_mu_nominal = 1.0
    track_mu = 1.0

    # Set the list of *track* μ values you want to test in the SIMULATOR (not in MPC).
    mu_start = 0.6        # initial factor
    mu_end = 1.6        # final factor
    mu_step = 0.2        # step

    mpc_mu_values = [factor
                     for factor in np.arange(mu_start, mu_end + 1e-9, mu_step)]

    all_abs_errs = {}

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
        x_pred_seq_hist = []

        # Change the mu of the track only for simulating the dynamics
        mpc.set_sim_mu(track_mu)

        start_time = time.time()
        for k in range(iterations):
            # 1) Pin solver's initial state to current plant (first 8 states only)
            x0_solver = x_plant[:mpc.model.n_x].copy()
            mpc.acados_solver.set(0, "lbx", x0_solver)
            mpc.acados_solver.set(0, "ubx", x0_solver)

            # Set online params along horizon (unchanged here)
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
                    mu_mpc
                ], dtype=np.float64)
                mpc.acados_solver.set(i, "p", online_parameters)

            # 2) Solve MPC
            status = mpc.acados_solver.solve()

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

            # timing print
            elapsed_time = time.time() - start_time
            # print(f"[μ={mu_track:.2f} | {k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")
            start_time = time.time()

        x_hist = np.asarray(x_hist)
        u_hist = np.asarray(u_hist)
        x_pred_seq_hist = np.asarray(x_pred_seq_hist)

        accel_cur = x_hist[:-1, 7]
        accel_nxt = x_hist[1:, 7]
        delta_cur = x_hist[:-1, 5]
        delta_nxt = x_hist[1:, 5]
        jerk_for_step = u_hist[1:, 0]     # control used for x[k-1] -> x[k]
        ddelta_for_step = u_hist[1:, 1]

        accel_resid = accel_nxt - (accel_cur + jerk_for_step * dt)
        delta_resid = wrap_to_pi(delta_nxt - (delta_cur + ddelta_for_step * dt))

        print(f"[μ={mu_mpc:.2f}] mean|accel_resid|={np.mean(np.abs(accel_resid)):.3e}, "
              f"mean|delta_resid|={np.mean(np.abs(delta_resid)):.3e}, "
              f"max|accel_resid|={np.max(np.abs(accel_resid)):.3e}, "
              f"max|delta_resid|={np.max(np.abs(delta_resid)):.3e}")

        # -------------------------------
        # NEW: model-mismatch (one lap)
        # -------------------------------
        abs_errs, mean_abs_err = compute_mismatch_abs_errors_one_lap(
            x_hist=x_hist,
            x_pred_seq_hist=x_pred_seq_hist,
            track_len=track_len,
            nx=nx,
            angle_idx=ANGLE_IDX,
            s0_init=float(init_pose_frenet[0])  # <-- initial s of the very first horizon
        )

        abs_errs = abs_errs[:, 1:]   # <--- drop 's' column
        all_abs_errs[mu_mpc] = abs_errs

        # 10-dim "plant" state used by _dynamics_of_car:
        x_plant = np.array([
            init_pose_frenet[0], init_pose_frenet[1], init_pose_frenet[2],
            vx_initial, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ], dtype=float)

        # Reset simulation variables
        s0 = x_plant[0]
        mpc.mpc_sd = np.zeros((mpc.stmpc_config.N + 1, 2))

        mpc.mpc_init_params()
        mpc.apply_warm_start(pose_frenet=init_pose_frenet)

        # Pin stage-0 to the current plant state (you already do lbx/ubx)
        mpc.acados_solver.set(0, "x", x_plant[:nx])

        # ---- Metrics for this μ ----
        lt = compute_lap_time(x_hist, dt, track_len, idx_s=0)
        mad_n = compute_mean_abs_lateral_dev(x_hist, idx_n=1)
        mae_v = compute_mean_abs_speed_error(waypoints, x_hist, track_len, idx_s=0, idx_vx=3)

    state_labels = ["n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]

    # --- Manual normalization ---
    # Define your scales per state label (units must match your states: radians for angles!)
    user_scales = {
        "n": 0.025,             # meters
        "theta": np.deg2rad(0.7),   # 5 deg -> radians
        "v_x": 0.05,           # m/s
        "v_y": 0.4,           # m/s
        "delta": np.deg2rad(0.3),   # 2 deg -> radians
        "yaw_rate": 0.5,      # rad/s (example)
        "accel": 0.05          # m/s^2
    }

    scales_vec = build_scale_vector_from_user(user_scales, state_labels)
    all_abs_errs_norm = apply_normalization_manual(all_abs_errs, scales_vec, clip_at_one=False)

    # Optional: drop 's' from visualization (common when spatial alignment makes s=0 by construction)
    # idx_s = state_labels.index("s")
    # for mu in list(all_abs_errs_norm.keys()):
    #     all_abs_errs_norm[mu] = np.delete(all_abs_errs_norm[mu], idx_s, axis=1)
    # state_labels_plot = [lbl for lbl in state_labels if lbl != "s"]
    # else:
    state_labels_plot = [f"{lbl} (manual)" for lbl in state_labels]

    fig_all = plot_state_error_boxplot_all(
        all_abs_errs_norm,
        state_labels=state_labels_plot,
        title="Normalized model mismatch per state (one lap, all μ) — manual scales",
        ylabel="Normalized absolute error (unitless)"
    )
    plt.tight_layout()
    plt.show()
