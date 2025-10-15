from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from f110_msgs.msg import Wpnt
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from single_track_mpc.STMPC import STMPC

# -----------------------
# Helpers (compact, core)
# -----------------------


def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    return np.array([(wp.x_m, wp.y_m, wp.vx_mps, wp.kappa_radpm, wp.s_m, wp.d_left, wp.d_right)
                     for wp in waypoints], dtype=float)


def rk4_step(mpc, x: np.ndarray, dt: float) -> np.ndarray:
    def f(xx): return np.array(mpc._dynamics_of_car(0.0, xx), dtype=float)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _unwrap_s_progress(s_seq: np.ndarray, track_len: float) -> np.ndarray:
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


def nearest_wp_idx_by_s(waypoints_s: np.ndarray, s_val: float, track_len: float) -> int:
    s_mod = s_val % track_len
    return int(np.abs(waypoints_s - s_mod).argmin())


def compute_ax_ay_from_hist(x_hist: np.ndarray,
                            waypoints: np.ndarray,
                            track_len: float,
                            idx_s: int = 0,
                            idx_vx: int = 3,
                            dt: float = 0.025) -> Tuple[np.ndarray, np.ndarray]:
    """ax ≈ dvx/dt (finite diff). ay ≈ vx^2 * kappa(s)."""
    s_seq = x_hist[:, idx_s]
    vx = x_hist[:, idx_vx]
    ax = np.empty_like(vx)
    ax[1:-1] = (vx[2:] - vx[:-2]) / (2 * dt)
    ax[0] = (vx[1] - vx[0]) / dt
    ax[-1] = (vx[-1] - vx[-2]) / dt

    wp_s = waypoints[:, 4]
    wp_kappa = waypoints[:, 3]
    idxs = np.abs(wp_s[None, :] - (s_seq % track_len)[:, None]).argmin(axis=1)
    kappa = wp_kappa[idxs]
    ay = vx**2 * kappa
    return ax, ay


def try_get_stage_bounds(acados_solver, N: int) -> Dict[str, Optional[np.ndarray]]:
    """Attempt to read solver bounds; if not available, return None for those."""
    keys = {"lbx": None, "ubx": None, "lbu": None, "ubu": None}
    for k in keys.keys():
        try:
            keys[k] = np.array(acados_solver.get(0, k), dtype=float)
        except Exception:
            keys[k] = None
    sizes = {}
    if keys["lbx"] is not None:
        sizes["nx"] = keys["lbx"].shape[0]
    if keys["lbu"] is not None:
        sizes["nu"] = keys["lbu"].shape[0]
    return {"bounds": keys, "sizes": sizes}


def get_input_bound_fallbacks(stmpc_config: STMPCConfig) -> Dict[str, Optional[Tuple[float, float]]]:
    def pick(lo_name, up_name):
        lo = getattr(stmpc_config, lo_name, None)
        up = getattr(stmpc_config, up_name, None)
        if lo is None or up is None:
            return None
        return (float(lo), float(up))
    return {
        "jerk": pick("jerk_min", "jerk_max"),
        "ddelta": pick("ddelta_min", "ddelta_max"),
    }


def count_hits(values: np.ndarray, lo: Optional[float], hi: Optional[float],
               tol: float) -> Tuple[int, int]:
    if lo is None and hi is None:
        return (0, 0)
    lo_hits = int(np.sum(np.isfinite(values) & (lo is not None) &
                         (np.abs(values - lo) <= tol)))
    hi_hits = int(np.sum(np.isfinite(values) & (hi is not None) &
                         (np.abs(values - hi) <= tol)))
    return lo_hits, hi_hits

# -----------------------
# Main (μ sweep + logs)
# -----------------------


if __name__ == "__main__":
    import time

    # --- Load track & configs ---
    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")
    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts
    waypoints = extract_waypoints_data(refline)

    racecar_version = "NUC5"
    floor = "dubi"
    init_pose_frenet = np.array([20.0, 0.0, 0.0])
    vx_initial = 3.0

    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config = load_trailing_config_ros(racecar_version)

    mpc = STMPC(
        pose_frenet=init_pose_frenet,
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

    # μ parameters
    track_mu = 1.0  # used INSIDE MPC model
    mu_start, mu_end, mu_step = 0.6, 1.6, 0.2
    mpc_mu_values = [float(x) for x in np.arange(mu_start, mu_end + 1e-12, mu_step)]

    # Simulation settings
    iterations = 500
    integration = "rk4"  # or "euler"

    # Tolerances
    HIT_TOL = 1e-4
    KIN_TOL = 0.05

    # Solver bounds + fallbacks
    solver_info = try_get_stage_bounds(mpc.acados_solver, stmpc_config.N)
    u_fallbacks = get_input_bound_fallbacks(stmpc_config)

    # Results per μ
    results_constraints: Dict[float, Dict[str, Dict[str, int]]] = {}
    gg_data: Dict[float, Dict[str, np.ndarray]] = {}

    # ---- NEW: prepare a single GG figure for all μ ----
    fig_all, ax_all = plt.subplots(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(mpc_mu_values)))
    # limits will be set after collecting all data
    global_max_abs = 0.0

    # Kinematic limits (constant across μ; read once)
    a_lat_max_conf = float(stmpc_config.alat_max)
    a_min_conf = float(stmpc_config.a_min)
    a_max_conf = float(stmpc_config.a_max)

    # 10-dim plant state
    def make_init_state():
        return np.array([
            init_pose_frenet[0], init_pose_frenet[1], init_pose_frenet[2],
            vx_initial, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ], dtype=float)

    for mu_idx, mu_mpc in enumerate(mpc_mu_values):
        print(f"\n=== μ_mpc = {mu_mpc:.2f} (Δμ={mu_mpc - track_mu:+.2f}) ===")
        mpc.set_sim_mu(track_mu)

        # Reset plant + MPC
        x_plant = make_init_state()
        mpc.mpc_sd = np.zeros((stmpc_config.N + 1, 2))
        mpc.mpc_init_params()
        mpc.apply_warm_start(pose_frenet=init_pose_frenet)

        u_hist = []
        x_hist = []
        u_stage_hits = {"jerk": {"lb": 0, "ub": 0}, "ddelta": {"lb": 0, "ub": 0}}
        x_bound_hits = {}

        lbx = solver_info["bounds"]["lbx"]
        ubx = solver_info["bounds"]["ubx"]
        nx = mpc.model.n_x
        if lbx is not None and ubx is not None:
            for i in range(nx):
                x_bound_hits[i] = {"lb": 0, "ub": 0}

        start = time.time()
        for k in range(iterations):
            # Pin stage-0 to plant
            x0 = x_plant[:mpc.model.n_x].copy()
            mpc.acados_solver.set(0, "lbx", x0)
            mpc.acados_solver.set(0, "ubx", x0)
            mpc.acados_solver.set(0, "x", x0)

            # Online params along horizon
            for i in range(stmpc_config.N + 1):
                idx = nearest_wp_idx_by_s(waypoints[:, 4], mpc.mpc_sd[i, 0], track_len)
                target_v_speed = waypoints[idx, 2]
                p = np.array([
                    target_v_speed,
                    stmpc_config.qadv, stmpc_config.qv, stmpc_config.qn, stmpc_config.qalpha,
                    stmpc_config.qjerk, stmpc_config.qddelta,
                    stmpc_config.alat_max, stmpc_config.a_min, stmpc_config.a_max,
                    stmpc_config.track_safety_margin,
                    0,
                    mu_mpc
                ], dtype=np.float64)
                mpc.acados_solver.set(i, "p", p)

            status = mpc.acados_solver.solve()
            if not (status == 0 or status == 2):
                mpc.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])

            # u0 = [jerk, ddelta]
            u0 = mpc.acados_solver.get(0, "u").astype(float)
            u_hist.append(u0.copy())

            # Count input bound hits
            lbu = solver_info["bounds"]["lbu"]
            ubu = solver_info["bounds"]["ubu"]

            def hit_count_one(val, lb, ub, fb_key):
                lo = lb if lb is not None else (u_fallbacks[fb_key][0] if u_fallbacks[fb_key] else None)
                hi = ub if ub is not None else (u_fallbacks[fb_key][1] if u_fallbacks[fb_key] else None)
                lb_hit = int(lo is not None and abs(val - lo) <= HIT_TOL)
                ub_hit = int(hi is not None and abs(val - hi) <= HIT_TOL)
                return lb_hit, ub_hit

            # jerk
            lbj = lbu[0] if lbu is not None and lbu.size >= 1 else None
            ubj = ubu[0] if ubu is not None and ubu.size >= 1 else None
            lb_hit, ub_hit = hit_count_one(u0[0], lbj, ubj, "jerk")
            u_stage_hits["jerk"]["lb"] += lb_hit
            u_stage_hits["jerk"]["ub"] += ub_hit

            # ddelta
            lbd = lbu[1] if lbu is not None and lbu.size >= 2 else None
            ubd = ubu[1] if ubu is not None and ubu.size >= 2 else None
            lb_hit, ub_hit = hit_count_one(u0[1], lbd, ubd, "ddelta")
            u_stage_hits["ddelta"]["lb"] += lb_hit
            u_stage_hits["ddelta"]["ub"] += ub_hit

            # Optional state bound hits at stage 0
            if lbx is not None and ubx is not None:
                x_stage0 = mpc.acados_solver.get(0, "x").astype(float)
                for i in range(min(nx, x_stage0.size)):
                    if np.isfinite(lbx[i]) and abs(x_stage0[i] - lbx[i]) <= HIT_TOL:
                        x_bound_hits[i]["lb"] += 1
                    if np.isfinite(ubx[i]) and abs(x_stage0[i] - ubx[i]) <= HIT_TOL:
                        x_bound_hits[i]["ub"] += 1

            # Integrate plant
            x_plant[8] = u0[0]
            x_plant[9] = u0[1]
            if integration == "rk4":
                x_plant = rk4_step(mpc, x_plant, dt)
            else:
                xdot = np.array(mpc._dynamics_of_car(0.0, x_plant), dtype=float)
                x_plant = x_plant + dt * xdot

            x_hist.append(x_plant.copy())

        x_hist = np.asarray(x_hist)
        u_hist = np.asarray(u_hist)

        # -----------------------------
        # GG diagram data for this μ
        # -----------------------------
        ax_arr, ay_arr = compute_ax_ay_from_hist(x_hist, waypoints, track_len,
                                                 idx_s=0, idx_vx=3, dt=dt)

        # Kinematic hits (optional)
        kin_hits = {
            "ax_min": int(np.sum(ax_arr <= a_min_conf + KIN_TOL)),
            "ax_max": int(np.sum(ax_arr >= a_max_conf - KIN_TOL)),
            "ay_max": int(np.sum(np.abs(ay_arr) >= a_lat_max_conf - KIN_TOL))
        }

        # Store results
        results_constraints[mu_mpc] = {
            "u_hits": u_stage_hits,
            "x_hits": x_bound_hits,
            "kin_hits": kin_hits
        }
        gg_data[mu_mpc] = {"ax": ax_arr, "ay": ay_arr}

        # ---- NEW: add this μ to the shared figure ----
        c = colors[mu_idx]
        ax_all.scatter(ay_arr, ax_arr, s=6, alpha=0.6, label=f"μ_mpc={mu_mpc:.2f}", color=c)

        # Track global limits
        if ax_arr.size > 0 or ay_arr.size > 0:
            local_max = max(
                abs(a_min_conf), abs(a_max_conf), a_lat_max_conf,
                np.max(np.abs(ax_arr)) if ax_arr.size else 0.0,
                np.max(np.abs(ay_arr)) if ay_arr.size else 0.0
            )
            global_max_abs = max(global_max_abs, local_max)

        print(f"μ={mu_mpc:.2f} | input hits={u_stage_hits} | kinematic hits={kin_hits}")

    # ---- NEW: finalize the single GG plot (no saving to image) ----
    lim = global_max_abs * 1.1 if global_max_abs > 0 else 1.0
    ax_all.axhline(a_min_conf, linestyle='--', linewidth=1, color='k')
    ax_all.axhline(a_max_conf, linestyle='--', linewidth=1, color='k')
    ax_all.axvline(+a_lat_max_conf, linestyle='--', linewidth=1, color='k')
    ax_all.axvline(-a_lat_max_conf, linestyle='--', linewidth=1, color='k')
    ax_all.set_xlim(-lim, lim)
    ax_all.set_ylim(-lim, lim)
    ax_all.set_xlabel("Longitudinal acceleration $a_x$ [m/s²]")
    ax_all.set_ylabel("Lateral acceleration $a_y$ [m/s²]")
    ax_all.set_title("GG diagram across μ_track sweep")
    ax_all.grid(True, linestyle=':')
    ax_all.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()  # <--- produces a plot; does NOT save an image

    # -----------------------
    # Print summary
    # -----------------------
    # print("\n===== Constraint Analysis Summary =====")
    # for mu, res in results_constraints.items():
    #     print(f"μ_track={mu:.2f}")
    #     print("  Input hits:")
    #     for k, v in res["u_hits"].items():
    #         print(f"    {k}: lb={v['lb']}, ub={v['ub']}")
    #     print("  Kinematic hits:")
    #     for k, v in res["kin_hits"].items():
    #         print(f"    {k}: {v}")
    #     if res["x_hits"]:
    #         print("  State hits:")
    #         for idx, v in res["x_hits"].items():
    #             print(f"    state[{idx}]: lb={v['lb']}, ub={v['ub']}")
    #     print("-------------------------------------")
