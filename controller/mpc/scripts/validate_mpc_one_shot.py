# validate_mpc_one_shot.py
from typing import List, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from f110_msgs.msg import WpntArray, Wpnt

from plotting_utils import plot_centerline_with_lr  # (optional, not required)
from pbl_config import (
    CarConfig, PacejkaTireConfig, STMPCConfig, TrailingConfig,
    load_STMPC_config_ros, load_car_config_ros, load_pacejka_tire_config_ros, load_trailing_config_ros
)
from single_track_mpc.acados_settings import acados_settings
from frenet_converter.frenet_converter import FrenetConverter


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
                       centerline_xy[0]  - centerline_xy[-1]))  # closed loop
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
    idx = np.clip(idx, 0, N-1)
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
    return x10 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


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
    left_b  = centerline_xy + left_n * left_dist[:, None]
    right_b = centerline_xy - left_n * right_dist[:, None]

    # Convert Frenet (s,n) to XY for both trajectories
    pred_xy    = frenet_to_xy_batch(centerline_xy, s_ref, x_pred[:, [idx_s, idx_n]])
    rollout_xy = frenet_to_xy_batch(centerline_xy, s_ref, x_rollout[:, [idx_s, idx_n]])

    # Figure
    import math
    nrows = int(math.ceil(nx / ncols))
    fig = plt.figure(figsize=(14, 4 + 3.2 * nrows))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=nrows+1, ncols=ncols, figure=fig, wspace=0.35, hspace=0.6)

    # Left (big): track
    axL = fig.add_subplot(gs[0, :2]) if ncols >= 3 else fig.add_subplot(gs[0, :])
    axL.plot(centerline_xy[:, 0], centerline_xy[:, 1], lw=2, label='Centerline')
    axL.plot(left_b[:, 0], left_b[:, 1],  label='Left boundary')
    axL.plot(right_b[:, 0], right_b[:, 1], label='Right boundary')

    # Direction arrow
    axL.arrow(centerline_xy[0, 0], centerline_xy[0, 1],
              tang[0, 0], tang[0, 1], head_width=0.5, head_length=0.5)

    axL.plot(pred_xy[:, 0], pred_xy[:, 1], 'o-', ms=4, label='Predicted (solver)', zorder=5)
    axL.plot(rollout_xy[:, 0], rollout_xy[:, 1], 's--', ms=3, label='Simulated (rollout)', zorder=6)

    axL.set_title("Track & horizon path: predicted vs simulated")
    axL.set_xlabel("X (m)"); axL.set_ylabel("Y (m)")
    axL.axis("equal"); axL.grid(True); axL.legend(loc='best')

    # Right: per-state overlay
    # build axes grid below
    axes = []
    for r in range(1, nrows+1):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs[r, c]))
    t = np.arange(H+1)

    for i in range(nx):
        ax = axes[i]
        ax.plot(t, x_pred[:, i],    lw=2, label='pred')
        ax.plot(t, x_rollout[:, i], lw=1.5, linestyle='--', label='sim')
        ax.set_title(state_names[i]); ax.set_xlabel("k (stage)"); ax.set_ylabel("value")
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
    left_dist  = np.array([p.d_left for p in refline])
    right_dist = np.array([p.d_right for p in refline])
    waypoints  = extract_waypoints_data(refline)

    # Configs
    racecar_version = "NUC5"
    floor = "dubi"
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config:  CarConfig    = load_car_config_ros(racecar_version)
    tire_config: PacejkaTireConfig = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config: TrailingConfig = load_trailing_config_ros(racecar_version)
    
    # Simu params
    rk4 = True

    # Build solver
    # waypoint columns: [x, y, vx_ref, kappa, s, d_left, d_right]
    x = waypoints[:, 0]; y = waypoints[:, 1]
    kapparef = waypoints[:, 3]
    s_ref = waypoints[:, 4]
    d_left = waypoints[:, 5]; d_right = waypoints[:, 6]

    # initialize acados
    constraint, model, acados_solver, model_params = acados_settings(
        s_ref, kapparef, d_left, d_right,
        stmpc_config, car_config, tire_config
    )

    H  = stmpc_config.N
    nx = model.n_x
    nu = model.n_u
    dt = 1.0 / stmpc_config.MPC_freq

    # Initial Frenet pose [s, n, theta]
    pose_frenet = np.array([0.0, 0.0, 0.0], dtype=float)

    # Warm start (simple rollout with const accel & steer-rate)
    warm = np.zeros((H+1, nx + nu))
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
        for i in range(1, H+1):
            x10 = rk4_step_x10(x10, const_u, dt)
            warm[i, :nx] = x10[:nx]
            if i < H:
                warm[i, nx:] = const_u
    else:
        for i in range(1, H+1):
            x10[8] = const_u[1]  # derDelta
            x10[9] = const_u[0]  # jerk
            x10 = x10 + dt * f_expl_10(x10)
            warm[i, :nx] = x10[:nx]
            if i < H:
                warm[i, nx:] = const_u

    # load warm start to solver
    for i in range(H+1):
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
    x_pred = np.zeros((H+1, nx))
    u_pred = np.zeros((H,   nu))
    for i in range(H+1):
        x_pred[i, :] = acados_solver.get(i, "x")
        if i < H:
            u_pred[i, :] = acados_solver.get(i, "u")

    # ---- Roll out the nonlinear dynamics using u_pred ----
    # Build 10-dim "plant" vector used by f_expl (first 8 are model states)
    x_roll = np.zeros((H+1, nx))
    x10 = np.concatenate([x_pred[0, :], np.zeros(2)])  # append [derDelta, jerk]
    x_roll[0, :] = x10[:nx]

    if (rk4):
        for i in range(H):
            x10 = rk4_step_x10(x10, u_pred[i, :], dt)  # u_pred[i] = [jerk, derDelta]
            x_roll[i+1, :] = x10[:nx]
    else:
        for i in range(H):
            # set helper states from control
            x10[8] = u_pred[i, 0]  # jerk     = d(accel)/dt
            x10[9] = u_pred[i, 1]  # derDelta = d(delta)/dt

            # integrate one step
            x10 = x10 + dt * f_expl_10(x10)
            x_roll[i+1, :] = x10[:nx]

    # ---- Plot: predicted vs simulated ----
    names = ["s","n","theta","v_x","v_y","delta","yaw_rate","accel"]
    fig = plot_prediction_validation(
        centerline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        s_ref=_arc_length(refline_xy),  # compute arc-length (closed)
        x_pred=x_pred,                   # (H+1,nx)
        x_rollout=x_roll,                # (H+1,nx)
        state_names=names[:nx],
        suptitle="One-shot validation: solver prediction vs rollout with solver inputs"
    )
    plt.show()
