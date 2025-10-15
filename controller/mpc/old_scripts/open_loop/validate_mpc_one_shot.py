# validate_mpc_one_shot.py
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from f110_msgs.msg import Wpnt
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        load_car_config_ros, load_pacejka_tire_config_ros,
                        load_STMPC_config_ros)
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from single_track_mpc.acados_settings import acados_settings

# ---------------------------
# Helpers
# ---------------------------


def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    """Convert WpntArray to numpy array for easier processing.
       columns: [x, y, vx_ref, kappa, s, d_left, d_right]
    """
    waypoints_list = [(wpnt.x_m, wpnt.y_m, wpnt.vx_mps, wpnt.kappa_radpm, wpnt.s_m, wpnt.d_left, wpnt.d_right)
                      for wpnt in waypoints]
    return np.array(waypoints_list)


def _compute_unit_tangents(centerline_xy: np.ndarray) -> np.ndarray:
    diffs = np.vstack((centerline_xy[1:] - centerline_xy[:-1],
                       centerline_xy[0] - centerline_xy[-1]))  # closed loop
    tang = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)
    return tang


def _compute_unit_left_normals(tangents: np.ndarray) -> np.ndarray:
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _arc_length(centerline_xy: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(centerline_xy, axis=0, append=centerline_xy[:1]), axis=1)
    s = np.cumsum(seg)
    s = np.insert(s, 0, 0.0)  # length N+1 with s[-1] = track length
    return s


def _wrap_s(s: np.ndarray, L: float) -> np.ndarray:
    return np.mod(s, L)


def frenet_to_xy_batch(centerline_xy: np.ndarray,
                       s_ref: np.ndarray,
                       sn_batch: np.ndarray) -> np.ndarray:
    """Convert (s,n) batch (Q,2) to XY along closed centerline."""
    N = centerline_xy.shape[0]
    L = s_ref[-1]
    s = _wrap_s(sn_batch[:, 0], L)
    n = sn_batch[:, 1]

    idx = np.searchsorted(s_ref, s, side='right') - 1
    idx = np.clip(idx, 0, N - 1)
    idx_next = (idx + 1) % N

    s_i = s_ref[idx]
    ds_i = s_ref[idx_next] - s_ref[idx]
    ds_i = np.where(ds_i <= 0.0, ds_i + L, ds_i)

    alpha = np.where(ds_i > 1e-12, (s - s_i) / ds_i, 0.0)
    p_i = centerline_xy[idx]
    p_j = centerline_xy[idx_next]
    p = p_i + (p_j - p_i) * alpha[:, None]

    tang = _compute_unit_tangents(centerline_xy)
    left_n = _compute_unit_left_normals(tang)
    n_vec = left_n[idx]
    xy = p + n[:, None] * n_vec
    return xy

# ---- RK4 helper (keeps u = [jerk, derDelta] constant over the step) ----


def rk4_step_x10(x10: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    # u = [jerk, derDelta]
    def f(x):
        xx = x.copy()
        xx[8] = u[0]   # jerk
        xx[9] = u[1]   # derDelta
        return f_expl_10(xx)  # returns xdot in the same 10-dim layout

    k1 = f(x10)
    k2 = f(x10 + 0.5 * dt * k1)
    k3 = f(x10 + 0.5 * dt * k2)
    k4 = f(x10 + dt * k3)
    return x10 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def f_expl_10_with_p(x10, p_vec):
    s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta = x10
    xdot = model.f_expl_func(s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta, p_vec)
    return np.array([float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
                     float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]),
                     0.0, 0.0], dtype=float)


# --- helpers for spatial interpolation (drop these near your other helpers) ---


def _unwrap_monotone_s(s_nodes: np.ndarray, track_len: float) -> np.ndarray:
    """Return a strictly increasing copy of s_nodes by adding multiples of track_len after wrap."""
    s_un = [float(s_nodes[0])]
    for i in range(1, len(s_nodes)):
        s_next = float(s_nodes[i])
        if s_next < s_nodes[i - 1]:
            s_next += track_len
        s_un.append(s_next)
    return np.array(s_un, dtype=float)


def _extend_for_wrap(s_un: np.ndarray, arr: np.ndarray, track_len: float):
    """
    Make arrays periodic to allow interpolation past the last node:
      returns (s_ext, arr_ext) where the last point repeats the first + track_len.
    - s_un: (H+1,)
    - arr:  (H+1, D) for states/params, (H, D) for inputs (handle below)
    """
    s_ext = np.concatenate([s_un, [s_un[0] + track_len]])
    arr_ext = np.concatenate([arr, arr[:1]], axis=0)
    return s_ext, arr_ext


def _interp_along_s(sq: float, s_ext: np.ndarray, v_ext: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear interpolation v(s) at scalar sq.
    s_ext: (K,), strictly increasing; v_ext: (K, D) matches s_ext.
    """
    # locate interval [i, i+1] with s_ext[i] <= sq < s_ext[i+1]
    i = np.searchsorted(s_ext, sq, side="right") - 1
    i = np.clip(i, 0, len(s_ext) - 2)
    s0, s1 = s_ext[i], s_ext[i + 1]
    if s1 <= s0 + 1e-12:
        return v_ext[i].copy()
    alpha = (sq - s0) / (s1 - s0)
    return (1.0 - alpha) * v_ext[i] + alpha * v_ext[i + 1]


def rk4_step_x10_with_p(x10: np.ndarray, u: np.ndarray, dt: float, p_vec) -> np.ndarray:
    """RK4 with constant (jerk, derDelta) over step and explicit params p_vec."""
    def f(x):
        xx = x.copy()
        xx[8] = u[0]     # jerk
        xx[9] = u[1]     # derDelta
        return f_expl_10_with_p(xx, p_vec)  # uses your model.f_expl_func(..., p_vec)
    k1 = f(x10)
    k2 = f(x10 + 0.5 * dt * k1)
    k3 = f(x10 + 0.5 * dt * k2)
    k4 = f(x10 + dt * k3)
    return x10 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# 0) Utilities for time span from spatial horizon


def _unwrap_ds_sequence(s_nodes: np.ndarray, track_len: float) -> np.ndarray:
    ds = np.diff(s_nodes)
    return np.where(ds < 0.0, ds + track_len, ds)


def _sdot_at_node(x_node, u_node):
    # Build a 10-dim vector for xdot evaluation with given u=[jerk, derDelta]
    xx = np.concatenate([x_node, [u_node[0], u_node[1]]])
    xdot = f_expl_10(xx)  # uses model_params.p inside
    return float(xdot[0])


def _map_s_to_period(s_val: float, base: float, L: float) -> float:
    # map scalar s_val to [base, base+L) without branching explosions
    y = s_val
    while y < base:
        y += L
    while y >= base + L:
        y -= L
    return y


def _zoh_along_s(sq: float, s_ext: np.ndarray, v_ext: np.ndarray) -> np.ndarray:
    """Zero-order hold sampling at sq: pick value at the left node of the interval containing sq."""
    i = np.searchsorted(s_ext, sq, side="right") - 1
    i = np.clip(i, 0, len(s_ext) - 2)
    return v_ext[i].copy()

# Fix helper derivatives for clarity: keep inputs constant inside the RK4 step


def f_expl_10_with_p(x10, p_vec):
    s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta = x10
    xdot = model.f_expl_func(s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta, p_vec)
    return np.array([
        float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
        float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]),
        0.0, 0.0   # keep helper slots constant within the step
    ], dtype=float)


# ---------------------------
# Plot: predicted vs simulated rollout over one horizon
# ---------------------------


def plot_prediction_validation(
    centerline_xy: np.ndarray,
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    s_ref: np.ndarray,
    x_pred: np.ndarray,         # (H+1, nx)
    x_rollout: np.ndarray,      # (H+1, nx) rollout of dynamics using u_pred
    *,
    state_names: Optional[Sequence[str]] = None,
    idx_s: int = 0,
    idx_n: int = 1,
    ncols: int = 3,
    suptitle: str = "One-shot validation: MPC prediction vs simulated rollout"
) -> plt.Figure:
    """
    Left: track + predicted XY path + simulated XY rollout.
    Right: per-state overlay over k=0..H.
    """
    assert x_pred.ndim == 2 and x_rollout.ndim == 2
    assert x_pred.shape == x_rollout.shape
    Hp1, nx = x_pred.shape
    H = Hp1 - 1

    if state_names is None or len(state_names) < nx:
        defaults = [f"x[{i}]" for i in range(nx)]
        state_names = defaults if state_names is None else list(state_names) + defaults[len(state_names):nx]

    # Track boundaries
    tang = _compute_unit_tangents(centerline_xy)
    left_n = _compute_unit_left_normals(tang)
    left_b = centerline_xy + left_n * left_dist[:, None]
    right_b = centerline_xy - left_n * right_dist[:, None]

    # Convert Frenet (s,n) to XY for both trajectories
    pred_xy = frenet_to_xy_batch(centerline_xy, s_ref, x_pred[:, [idx_s, idx_n]])
    rollout_xy = frenet_to_xy_batch(centerline_xy, s_ref, x_rollout[:, [idx_s, idx_n]])

    # Figure
    import math
    nrows = int(math.ceil(nx / ncols))
    fig = plt.figure(figsize=(14, 4 + 3.2 * nrows))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=nrows + 1, ncols=ncols, figure=fig, wspace=0.35, hspace=0.6)

    # Left (big): track
    axL = fig.add_subplot(gs[0, :2]) if ncols >= 3 else fig.add_subplot(gs[0, :])
    axL.plot(centerline_xy[:, 0], centerline_xy[:, 1], lw=2, label='Centerline')
    axL.plot(left_b[:, 0], left_b[:, 1], label='Left boundary')
    axL.plot(right_b[:, 0], right_b[:, 1], label='Right boundary')

    # Direction arrow
    axL.arrow(centerline_xy[0, 0], centerline_xy[0, 1],
              tang[0, 0], tang[0, 1], head_width=0.5, head_length=0.5)

    axL.plot(pred_xy[:, 0], pred_xy[:, 1], 'o-', ms=4, label='Predicted (solver)', zorder=5)
    axL.plot(rollout_xy[:, 0], rollout_xy[:, 1], 's--', ms=3, label='Simulated (rollout)', zorder=6)

    axL.set_title("Track & horizon path: predicted vs simulated")
    axL.set_xlabel("X (m)")
    axL.set_ylabel("Y (m)")
    axL.axis("equal")
    axL.grid(True)
    axL.legend(loc='best')

    # Right: per-state overlay
    # build axes grid below
    axes = []
    for r in range(1, nrows + 1):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))
    t = np.arange(H + 1)

    for i in range(nx):
        ax = axes[i]
        ax.plot(t, x_pred[:, i], lw=2, label='pred')
        ax.plot(t, x_rollout[:, i], lw=1.5, linestyle='--', label='sim')
        ax.set_title(state_names[i])
        ax.set_xlabel("k (stage)")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    # hide any unused small axes
    for j in range(nx, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------
# Main: one-shot validation
# ---------------------------

if __name__ == "__main__":
    # Load track
    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")
    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts
    refline_xy = np.array([(p.x_m, p.y_m) for p in refline])
    left_dist = np.array([p.d_left for p in refline])
    right_dist = np.array([p.d_right for p in refline])
    waypoints = extract_waypoints_data(refline)

    # Configs
    racecar_version = "NUC5"
    floor = "dubi"
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config: PacejkaTireConfig = load_pacejka_tire_config_ros(racecar_version, floor)

    # Simu params
    rk4 = True

    # Build solver
    # waypoint columns: [x, y, vx_ref, kappa, s, d_left, d_right]
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    kapparef = waypoints[:, 3]
    s_ref = waypoints[:, 4]
    d_left = waypoints[:, 5]
    d_right = waypoints[:, 6]

    # initialize acados
    constraint, model, acados_solver, model_params = acados_settings(
        s_ref, kapparef, d_left, d_right,
        stmpc_config, car_config, tire_config
    )

    H = stmpc_config.N
    nx = model.n_x
    nu = model.n_u
    dt = 1.0 / stmpc_config.MPC_freq

    # Initial Frenet pose [s, n, theta]
    pose_frenet = np.array([0.0, 0.0, 0.0], dtype=float)

    # Warm start (simple rollout with const accel & steer-rate)
    warm = np.zeros((H + 1, nx + nu))
    # our internal plant uses 10-dim but model has 8; we only seed model states (first nx)
    # state order: [s, n, theta, v_x, v_y, delta, yaw_rate, accel]
    warm[0, :nx] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2],
                             1.0, 0.0, 0.0, 0.0, 0.5])  # initial guess
    const_u = np.array([0.0, 0.0])  # [jerk, steer_rate]
    warm[0, nx:] = const_u

    # forward-Euler warm for states only
    def f_expl_10(x10):
        s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta = x10
        xdot = model.f_expl_func(s, n, th, vx, vy, delta, yaw_rate, accel, jerk, derDelta, model_params.p)
        # keep helper states as inputs
        return np.array([float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
                         float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]),
                         jerk, derDelta], dtype=float)

    x10 = np.concatenate([warm[0, :nx], const_u[::-1]])  # helper ordering: [jerk, derDelta] = [u[0], u[1]]

    if (rk4):
        for i in range(1, H + 1):
            x10 = rk4_step_x10(x10, const_u, dt)
            warm[i, :nx] = x10[:nx]
            if i < H:
                warm[i, nx:] = const_u
    else:
        for i in range(1, H + 1):
            x10[8] = const_u[0]  # jerk
            x10[9] = const_u[1]  # derDelta
            x10 = x10 + dt * f_expl_10(x10)
            warm[i, :nx] = x10[:nx]
            if i < H:
                warm[i, nx:] = const_u

    # load warm start to solver
    for i in range(H + 1):
        acados_solver.set(i, "x", warm[i, :nx])
        if i < H:
            acados_solver.set(i, "u", warm[i, nx:])

    # Pin initial state to the chosen x0
    x0 = warm[0, :nx].copy()
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

    # ---- One MPC solve ----
    status = acados_solver.solve()
    print(f"[validate] solver status = {status}")

    # Extract solver prediction (H+1,nx) and inputs (H,nu)
    x_pred = np.zeros((H + 1, nx))
    u_pred = np.zeros((H, nu))
    for i in range(H + 1):
        x_pred[i, :] = acados_solver.get(i, "x")
        if i < H:
            u_pred[i, :] = acados_solver.get(i, "u")

    try:
        p_nodes = [acados_solver.get(i, "p") for i in range(H + 1)]  # (H+1, n_p)
    except Exception:
        p_nodes = None

    # =======================
    # TIME ROLLOUT @ fixed dt (interpolate by s each step) + TIME-ALIGNED PLOTS
    # =======================

    # 1) Track length and unwrap solver s to make it strictly increasing
    track_len = s_ref[-1] if s_ref[-1] > 0 else (_arc_length(refline_xy)[-1])
    s_pred_un = _unwrap_monotone_s(x_pred[:, 0], track_len)      # (H+1,)

    # 2) Compute horizon time-span implied by spatial nodes: dt_k = ds_k / sdot_k
    ds_seq = _unwrap_ds_sequence(x_pred[:, 0], track_len)         # (H,)
    sdot_nodes = np.array([max(1e-6, _sdot_at_node(x_pred[k], u_pred[k])) for k in range(H)])  # clamp
    dt_nodes = ds_seq / sdot_nodes                                # (H,)
    T_horizon = float(np.sum(dt_nodes))

    # 3) Decide number of time steps so total span matches horizon span
    dt = 1.0 / stmpc_config.MPC_freq                               # fixed simulation dt
    N_t = max(1, int(round(T_horizon / dt)))                       # simulate ~same duration
    t_vec = np.linspace(0.0, N_t * dt, N_t + 1)                    # (N_t+1,)

    # 4) Build periodic (wrapped) arrays for interpolation by s
    #    States (H+1,nx), parameters per node (H+1,n_p, optional), and controls (H,nu)
    s_x_ext, x_ext = _extend_for_wrap(s_pred_un, x_pred, track_len)
    if p_nodes is not None:
        p_nodes_arr = np.asarray(p_nodes)                           # (H+1, n_p)
        s_p_ext, p_ext = _extend_for_wrap(s_pred_un, p_nodes_arr, track_len)
    else:
        s_p_ext, p_ext = s_x_ext, None

    # Controls: make node-aligned (H+1,nu) by wrapping the first at the end
    u_nodes = np.vstack([u_pred, u_pred[:1]])                      # (H+1, nu)
    s_u_ext, u_ext = _extend_for_wrap(s_pred_un, u_nodes, track_len)

    # 5) Time-stepped simulation
    x_roll_time = np.zeros((N_t + 1, nx))
    x10 = np.concatenate([x_pred[0, :], np.zeros(2)])              # start from solver x0; helper slots zero
    x_roll_time[0, :] = x10[:nx]

    base = s_pred_un[0]  # base period for mapping s to [base, base+track_len)

    INPUT_INTERP_MODE = "linear"   # "zoh" or "linear"

    u_used_time = np.zeros((N_t + 1, nu))
    sdot_time = np.zeros(N_t + 1)
    # initialize at t=0
    s0q = _map_s_to_period(float(x_roll_time[0, 0]), base, track_len)
    u_used_time[0, :] = (_zoh_along_s(s0q, s_u_ext, u_ext) if INPUT_INTERP_MODE == "zoh"
                         else _interp_along_s(s0q, s_u_ext, u_ext))[:nu]
    # record sdot at t=0
    xx0 = np.concatenate([x_roll_time[0, :], [u_used_time[0, 0], u_used_time[0, 1]]])
    sdot_time[0] = float(f_expl_10(xx0)[0])

    for k in range(N_t):
        # current s from simulated state
        s_cur = float(x10[0])
        s_query = _map_s_to_period(s_cur, base, track_len)

        # interpolate inputs (and params) at s_query
        if INPUT_INTERP_MODE == "zoh":
            u_here = _zoh_along_s(s_query, s_u_ext, u_ext)[:nu]
            p_here = (_zoh_along_s(s_query, s_p_ext, p_ext) if p_ext is not None else model_params.p)
        else:
            u_here = _interp_along_s(s_query, s_u_ext, u_ext)[:nu]
            p_here = (_interp_along_s(s_query, s_p_ext, p_ext) if p_ext is not None else model_params.p)

        # log inputs
        u_used_time[k + 1, :] = u_here

        # log sdot (for diagnostics)
        xx_tmp = np.concatenate([x10[:nx], [u_here[0], u_here[1]]])
        sdot_time[k + 1] = float(f_expl_10(xx_tmp)[0])

        # choose sub-steps to keep per-step changes small when sdot is small or inputs are large
        h_base = dt
        # heuristic: bound Δa ≈ |jerk|*h and Δδ ≈ |derDelta|*h
        J = abs(u_here[0])
        D = abs(u_here[1])
        S = abs(sdot_time[k]) + 1e-6
        M_sub = max(1,
                    int(np.ceil(J * h_base / 0.02)),    # keep accel change per substep ≤ 0.02 m/s^2
                    int(np.ceil(D * h_base / 0.002)),   # keep steering change per substep ≤ 0.002 rad
                    int(np.ceil(0.5 / (S * h_base)))    # if ṡ is small, increase resolution
                    )

        h = h_base / M_sub
        for _ in range(M_sub):
            x10 = rk4_step_x10_with_p(x10, u_here, h, p_here)

        # store state
        x_roll_time[k + 1, :] = x10[:nx]

    u_mpc_time = np.zeros_like(u_used_time)
    for k in range(N_t + 1):
        s_cur = float(x_roll_time[k, 0])
        s_query = _map_s_to_period(s_cur, base, track_len)
        if INPUT_INTERP_MODE == "zoh":
            u_mpc_time[k, :] = _zoh_along_s(s_query, s_u_ext, u_ext)[:nu]
        else:
            u_mpc_time[k, :] = _interp_along_s(s_query, s_u_ext, u_ext)[:nu]

    # 6) Build a TIME-ALIGNED MPC prediction by interpolating x_pred along s(t)
    x_pred_time = np.zeros_like(x_roll_time)
    for k in range(N_t + 1):
        s_cur = float(x_roll_time[k, 0])   # use simulated s for comparison point
        s_query = _map_s_to_period(s_cur, base, track_len)
        x_pred_time[k, :] = _interp_along_s(s_query, s_x_ext, x_ext)

    # 7) Plot states vs TIME (not stage)
    names = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]
    nx_plot = min(nx, len(names))

    err = x_roll_time - x_pred_time
    rmse = np.sqrt(np.mean(err**2, axis=0))
    print("Per-state RMSE:", dict(zip(names[:nx], rmse[:nx])))

    import math
    ncols = 3
    nrows = int(math.ceil(nx_plot / ncols))
    fig = plt.figure(figsize=(14, 4 + 3.2 * nrows))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, wspace=0.35, hspace=0.6)

    axes = []
    for r in range(nrows):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))

    for i in range(nx_plot):
        ax = axes[i]
        ax.plot(t_vec, x_pred_time[:, i], lw=2, label='MPC pred (interp @ s(t))')
        ax.plot(t_vec, x_roll_time[:, i], lw=1.5, linestyle='--', label='Model sim')
        ax.set_title(names[i])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.4)
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    # hide any unused subplots
    for j in range(nx_plot, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Time-based validation: MPC (interpolated by s) vs simulated model")
    fig.tight_layout()
    plt.show()

    # --- Inputs over time
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes2[0].plot(t_vec, u_used_time[:, 0], lw=1.7, label='jerk used')
    axes2[0].grid(True)
    axes2[0].legend()
    axes2[0].set_ylabel('jerk')
    axes2[1].plot(t_vec, u_used_time[:, 1], lw=1.7, label='derDelta used')
    axes2[1].grid(True)
    axes2[1].legend()
    axes2[1].set_ylabel('derDelta')
    axes2[2].plot(t_vec, sdot_time, lw=1.7, label='s_dot (sim)')
    axes2[2].grid(True)
    axes2[2].legend()
    axes2[2].set_ylabel('ṡ')
    axes2[2].set_xlabel('time (s)')
    fig2.suptitle(f'Inputs used by sim and ṡ(t) — sampling: {INPUT_INTERP_MODE.upper()}')
    fig2.tight_layout()

    plt.show()
