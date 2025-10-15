from typing import List

import matplotlib.pyplot as plt
import numpy as np
from f110_msgs.msg import Wpnt
from frenet_converter.frenet_converter import FrenetConverter
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
# plot raceline, adding arrow at first waypoint for general direction
from plotting_utils import (plot_closed_loop_summary,
                            plot_spaghetti_predictions_vs_actual,
                            plot_state_prediction_vs_actual,
                            plot_velocity_vs_progress)
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


if __name__ == "__main__":
    # Example of running the MPC solver
    import time

    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")

    # global_traj_wpnts_mintime.wpnts  # global_traj_wpnts_iqp.wpnts
    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts

    refline_xy = np.array([(p.x_m, p.y_m) for p in refline])

    left_dist = np.array([p.d_left for p in refline])

    right_dist = np.array([p.d_right for p in refline])

    # Car, track and mpc parameters
    racecar_version = "NUC5"
    floor = "dubi_hoons"
    pose_frenet = np.array([10.0, 0.0, 0.0])  # 0 position on the raceline
    vx_initial = 6.0
    waypoints = extract_waypoints_data(refline)
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config = load_trailing_config_ros(racecar_version)

    # Simulation config parameters
    iterations = 300  # TODO: Convert it to number of laps
    integration = "rk4"  # "euler" or "rk4"
    vis_spaghetti_predictions_vs_actual = False
    vis_state_prediction_vs_actual = False
    vis_closed_loop_summary = True
    vis_target_vel_actual = False
    track_mu = 1.0
    mpc_mu = 1.0

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

    # 10-dim "plant" state used by _dynamics_of_car:
    # [s, n, theta, vx, vy, delta, yaw_rate, accel, jerk, derDelta]
    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        vx_initial, 0.0, 0.0, 0.0, 0.0,   # vx, vy, delta, yaw_rate, accel
        0.0, 0.0                   # jerk, derDelta (helpers)
    ], dtype=float)

    x_hist = []
    u_hist = []
    status_hist = []
    x_pred1_hist = []   # predicted x at stage 1 (nx,)
    err_hist = []   # error vector (nx,)
    err_norm_hist = []   # ||error||
    x_pred_seq_hist = []

    # TEST FOR MU SENSITIVITY
    # Change the mu of the track
    mpc.set_sim_mu(track_mu)

    start_time = time.time()
    for k in range(iterations):
        # 1) Pin solver's initial state to current plant (first 8 states only)
        x0_solver = x_plant[:mpc.model.n_x].copy()
        mpc.acados_solver.set(0, "lbx", x0_solver)
        mpc.acados_solver.set(0, "ubx", x0_solver)

        for i in range(mpc.stmpc_config.N + 1):

            idx = np.abs(waypoints[:, 4] - mpc.mpc_sd[i, 0] % mpc.spline.track_length).argmin()
            target_v_speed = waypoints[idx, 2]
            online_parameters = np.array([target_v_speed,
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
                                          mpc_mu,
                                          stmpc_config.terminal_multiplier],
                                         dtype=np.float64)
            mpc.acados_solver.set(i, "p", online_parameters)

        # 2) Solve MPC
        status = mpc.acados_solver.solve()
        status_hist.append(status)

        if not (status == 0 or status == 2):  # 0 is sucess 2 is max iter, necesary for SQP
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
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
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        if (integration == "rk4"):
            x_plant = rk4_step(mpc, x_plant, dt)
        else:
            xdot = mpc._dynamics_of_car(0.0, x_plant)
            x_plant = x_plant + dt * np.array(xdot, dtype=float)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        elapsed_time = time.time() - start_time
        # if (k % 20) == 0:
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")
        start_time = time.time()

    x_pred_seq_hist = np.asarray(x_pred_seq_hist)
    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)

    if (vis_spaghetti_predictions_vs_actual):
        state_names = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]
        fig = plot_spaghetti_predictions_vs_actual(
            x_pred_seq_hist=x_pred_seq_hist,  # (K, H+1, nx)
            x_hist=x_hist,                    # (K, 10) -> uses first nx
            state_names=state_names[:x_pred_seq_hist.shape[2]],
            plot_every=1                      # increase (e.g., 3 or 5) if the plot is too dense
        )
        plt.show()

    if (vis_state_prediction_vs_actual):
        state_names = ["s", "n", "theta", "v_x", "v_y", "delta", "yaw_rate", "accel"]
        fig = plot_state_prediction_vs_actual(
            x_pred1_hist=x_pred_seq_hist[:, 1, :],
            x_hist=x_hist,
            state_names=state_names[:nx]
        )
        plt.show()

    if (vis_closed_loop_summary):
        fig = plot_closed_loop_summary(
            refline_xy=refline_xy,
            left_dist=left_dist,
            right_dist=right_dist,
            x_hist=x_hist,   # from your closed-loop simulation
            u_hist=u_hist
        )
        plt.show()

    if (vis_target_vel_actual):
        fig = plot_velocity_vs_progress(
            waypoints=waypoints,   # your Nx7 array
            x_hist=x_hist,         # your Kx10 array (or Kxnx), from the closed-loop run
            idx_s=0,
            idx_vx=3,
            title="Target vs Actual speed along s"
        )
        plt.show()
