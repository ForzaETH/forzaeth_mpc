#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from f110_msgs.msg import Wpnt
from learning_based_mpc.mpc_core.MPC.l4a_STMPC import STMPC_L4A
from learning_based_mpc.mpc_core.MPC.STMPC import STMPC
from learning_based_mpc.mpc_core.residual.gp_interface import (
    build_multitask_gp, gp_predictor_from_model, load_gp_state, save_gp_state,
    train_gp_model)
from learning_based_mpc.mpc_core.residual.residual_models import \
    make_residual_model_from_gp
from learning_based_mpc.mpc_core.utils.integrators import integrate_open_loop
# Fallback utilities if your local ones are not importable
from learning_based_mpc.validation.data.dataset_one_step import \
    build_dataset_one_step
from learning_based_mpc.validation.viz.plotting import plot_three
from old_scripts.plotting_utils import (plot_closed_loop_summary,
                                        plot_spaghetti_predictions_vs_actual)
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
# from learning_based_mpc.mpc_core.utils.readwrite_global_waypoints_no_ros import read_global_waypoints, Wpnt, GlobalWaypointsLite as GlobalWaypoints
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from single_track_mpc.utils.indicies import StateIndex

# from f110_msgs.msg import Wpnt


def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    return np.array([
        (wp.x_m, wp.y_m, wp.vx_mps, wp.kappa_radpm, wp.s_m, wp.d_left, wp.d_right)
        for wp in waypoints
    ], dtype=float)


def dyn_from_ctrl(ctrl: STMPC, mu_nom: float, bias=(0.0, 0.0, 0.05)):
    import casadi as ca
    x = ctrl.model.x
    u = ctrl.model.u
    p = ctrl.model.p
    f = ctrl.model.f_expl_expr
    f_fun = ca.Function("f_fun", [x, u, p], [f])
    pvec = np.array(ctrl.model_params.p, dtype=float).reshape(-1)
    pvec[12] = float(mu_nom)

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    b = np.array(bias, dtype=float)

    def dyn_nom(t, xv, uv):
        return np.array(f_fun(xv, uv, pvec)).reshape(-1)

    def dyn_act(t, xv, uv):
        dx = dyn_nom(t, xv, uv).copy()
        dx[idx_vx] += b[0]
        dx[idx_vy] += b[1]
        dx[idx_wz] += b[2]
        return dx

    return dyn_nom, dyn_act


def rk4_step_fxu(dyn, t, x, u, dt):
    k1 = dyn(t, x, u)
    k2 = dyn(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = dyn(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = dyn(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_open_loop_fxu(dyn, x0, U, dt, method="rk4"):
    X = [np.array(x0, dtype=float).copy()]
    t = 0.0
    for k in range(U.shape[0]):
        uk = U[k]
        if method.lower() == "rk4":
            x_next = rk4_step_fxu(dyn, t, X[-1], uk, dt)
        else:  # euler
            x_next = X[-1] + dt * dyn(t, X[-1], uk)
        X.append(x_next)
        t += dt
    return np.stack(X, axis=0)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gp_iters", type=int, default=300)
    ap.add_argument("--gp_lr", type=float, default=0.03)
    ap.add_argument("--bias_vx", type=float, default=1.0)
    ap.add_argument("--bias_vy", type=float, default=1.0)
    ap.add_argument("--bias_wz", type=float, default=1.0)
    ap.add_argument("--track", default="THETRACK")
    ap.add_argument("--racecar_version", default="NUC5")
    ap.add_argument("--floor", default="dubi_hoons")
    args = ap.parse_args()

    # === Waypoints & configs ===
    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")

    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts
    refline_xy = np.array([(p.x_m, p.y_m) for p in refline])
    left_dist = np.array([p.d_left for p in refline])
    right_dist = np.array([p.d_right for p in refline])
    waypoints = extract_waypoints_data(refline)

    stmpc_cfg: STMPCConfig = load_STMPC_config_ros(args.racecar_version)
    car_cfg: CarConfig = load_car_config_ros(args.racecar_version)
    tire_cfg: PacejkaTireConfig = load_pacejka_tire_config_ros(args.racecar_version, args.floor)
    trailing_cfg: TrailingConfig = load_trailing_config_ros(args.racecar_version)

    mu_nom = getattr(stmpc_cfg, "mu_nom", 1.0)
    dt = 1.0 / float(stmpc_cfg.MPC_freq)
    pose_frenet = np.array([10.0, 0.0, 0.0])
    init_vel = 3.0
    integration = "rk4"  # rk4 or euler

    # === 1) Nominal controller (STMPC) ===
    ctrl_nom = STMPC(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version=args.racecar_version,
        stmpc_config=stmpc_cfg,
        car_config=car_cfg,
        tire_config=tire_cfg,
        trailing_config=trailing_cfg,
        controller_frequency=float(stmpc_cfg.MPC_freq),
        using_gokart=False,
    )

    # Call the *same* API used on-vehicle
    # _ = ctrl_nom.main_loop(
    #     state="BGFree",
    #     position_in_map=np.zeros(3),
    #     waypoint_array_in_map=waypoints,
    #     speed_now=init_vel,
    #     opponent=None,
    #     position_in_map_frenet=pose_frenet,
    #     alpha=3.0,
    #     single_track_state=np.zeros(2),
    # )

    # Iterate with a simple plant to have a better mpc solution than the first one
    x_hist = []
    u_hist = []
    status_hist = []
    x_pred_seq_hist = []
    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        init_vel, 0.0, 0.0, 0.0, 0.0,   # vx, vy, delta, yaw_rate, accel
        0.0, 0.0                   # jerk, derDelta (helpers)
    ], dtype=float)

    for k in range(200):
        # 1) Pin solver's initial state to current plant (first 8 states only)
        x0_solver = x_plant[:ctrl_nom.model.n_x].copy()
        ctrl_nom.acados_solver.set(0, "lbx", x0_solver)
        ctrl_nom.acados_solver.set(0, "ubx", x0_solver)

        for i in range(ctrl_nom.stmpc_config.N + 1):

            idx = np.abs(waypoints[:, 4] - ctrl_nom.mpc_sd[i, 0] % ctrl_nom.spline.track_length).argmin()
            target_v_speed = waypoints[idx, 2]
            online_parameters = np.array([target_v_speed,
                                          stmpc_cfg.qadv,
                                          stmpc_cfg.qv,
                                          stmpc_cfg.qn,
                                          stmpc_cfg.qalpha,
                                          stmpc_cfg.qjerk,
                                          stmpc_cfg.qddelta,
                                          stmpc_cfg.alat_max,
                                          stmpc_cfg.a_min,
                                          stmpc_cfg.a_max,
                                          stmpc_cfg.track_safety_margin,
                                          0,
                                          mu_nom,
                                          stmpc_cfg.terminal_multiplier],
                                         dtype=np.float64)
            ctrl_nom.acados_solver.set(i, "p", online_parameters)

        # 2) Solve MPC
        status = ctrl_nom.acados_solver.solve()
        status_hist.append(status)

        if not (status == 0 or status == 2):  # 0 is sucess 2 is max iter, necesary for SQP
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
            ctrl_nom.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        # 3) Read optimal control (stage 0): u = [jerk, ddelta]
        u0 = ctrl_nom.acados_solver.get(0, "u").astype(float)
        u_hist.append(u0.copy())

        # Collect full horizon state predictions
        H = ctrl_nom.stmpc_config.N
        nx = ctrl_nom.model.n_x
        x_pred_full = np.zeros((H + 1, nx))
        for i in range(H + 1):
            x_pred_full[i, :] = ctrl_nom.acados_solver.get(i, "x")
        x_pred_seq_hist.append(x_pred_full)

        ctrl_nom.mpc_sd = np.array([
            ctrl_nom.acados_solver.get(j, "x")[:2] for j in range(ctrl_nom.stmpc_config.N + 1)
        ])

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        if (integration == "rk4"):
            x_plant = rk4_step(ctrl_nom, x_plant, dt)
        else:
            xdot = ctrl_nom._dynamics_of_car(0.0, x_plant)
            x_plant = x_plant + dt * np.array(xdot, dtype=float)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        print(f"[{k:04d}] status={status}, u0={u0}")

    # Visualize closed loop states and inputs
    x_pred_seq_hist = np.asarray(x_pred_seq_hist)
    x_hist = np.array(x_hist)
    u_hist = np.array(u_hist)
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=x_hist,
        u_hist=u_hist
    )
    plt.show()

    # Pull nominal trajectory from solver
    N = stmpc_cfg.N
    nx = ctrl_nom.model.n_x
    nu = ctrl_nom.model.n_u
    X_pred = np.stack([ctrl_nom.acados_solver.get(j, "x") for j in range(N + 1)], axis=0)
    U_pred = np.stack([ctrl_nom.acados_solver.get(j, "u") for j in range(N)], axis=0)

    # Visualize the last prediction horizon states and inputs
    U_aux = np.vstack([U_pred, U_pred[-1:]])
    X_aux = np.asarray(X_pred, dtype=float)
    U_aux = np.asarray(U_aux, dtype=float)
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=X_aux,   # from your closed-loop simulation
        u_hist=U_aux
    )
    plt.show()

    def write_u_into_state(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x[8] = u[0]     # jerk
        x[9] = u[1]     # derDelta
        return x

    # === 2) Integrate nominal vs actual ===
    dyn_nom, dyn_act = dyn_from_ctrl(ctrl_nom, mu_nom=float(mu_nom), bias=(args.bias_vx, args.bias_vy, args.bias_wz))
    X_nom = integrate_open_loop_fxu(dyn_nom, X_pred[0], U_pred, dt, method="rk4")

    # Visualize X_norm inegrated states
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=X_nom,
        u_hist=U_aux
    )
    plt.show()

    X_act = integrate_open_loop_fxu(dyn_act, X_pred[0], U_pred, dt, method="rk4")

    # Visualize X_act integrated states
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=X_act,
        u_hist=U_aux
    )
    plt.show()

    fig1, _ = plot_three(X_nom, X_act, None,
                         names=("vx", "vy", "wz"),
                         ref_xy=refline_xy,
                         track_left=left_dist,
                         track_right=right_dist,
                         idx_s=StateIndex.POS_ON_CENTER_LINE_S,   # <-- Frenet longitudinal coordinate
                         idx_n=StateIndex.MIN_DIST_TO_CENTER_LINE_N,   # <-- Frenet lateral coordinate
                         )

    # === 3) Dataset + 4) Train GP ===
    def nominal_step(x, u, DT): return integrate_open_loop_fxu(dyn_nom, x, u[None, :], DT, method="rk4")[-1]
    def actual_step(x, u, DT): return integrate_open_loop_fxu(dyn_act, x, u[None, :], DT, method="rk4")[-1]
    Z, Y = build_dataset_one_step(nominal_step, actual_step, X_pred, U_pred, dt)

    load_gp_model_from_state_dict = getattr(args, "load_gp", False)
    state_dict_path_gp_model = getattr(args, "gp_model_path", "gp_model_state_dict.pth")
    state_dict_path_likelihood = getattr(args, "gp_lik_path", "gp_model_likelihood_state_dict.pth")
    train_data_path = getattr(args, "gp_train_data_path", "gp_model_train_data.pth")  # optional

    device = args.device if hasattr(args, "device") else "cpu"
    training_iterations = getattr(args, "gp_iters", 200)
    rng_seed = getattr(args, "gp_seed", 456)
    lr = getattr(args, "gp_lr", 0.03)

    gp_model, likelihood = build_multitask_gp(Z, Y, device=device)

    if load_gp_model_from_state_dict:
        load_gp_state(gp_model, likelihood, state_dict_path_gp_model, state_dict_path_likelihood)
    else:
        gp_model, likelihood = train_gp_model(
            gp_model, likelihood, torch_seed=rng_seed,
            training_iterations=training_iterations, lr=lr, verbose=True
        )
        # save_gp_state(gp_model, likelihood, state_dict_path_gp_model, state_dict_path_likelihood)
        # try:
        #     import torch
        #     torch.save({"Z": torch.tensor(Z), "Y": torch.tensor(Y)}, train_data_path)
        # except Exception:
        #     pass

    # produce a numpy-based callable like before
    predict_mean = gp_predictor_from_model(gp_model, likelihood, device=device)

    # === 5) Residual-aware controller (STMPC_L4A) ===
    residual_model = make_residual_model_from_gp(gp_model, likelihood)

    ctrl_res = STMPC_L4A(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version=args.racecar_version,
        stmpc_config=stmpc_cfg,
        car_config=car_cfg,
        tire_config=tire_cfg,
        trailing_config=trailing_cfg,
        controller_frequency=float(stmpc_cfg.MPC_freq),
        using_gokart=False,
        residual_model=residual_model,
    )

    # _ = ctrl_res.main_loop(
    #     state="BGFree",
    #     position_in_map=np.zeros(3),
    #     waypoint_array_in_map=waypoints,
    #     speed_now=init_vel,
    #     opponent=None,
    #     position_in_map_frenet=pose_frenet,
    #     alpha=0.0,
    #     single_track_state=np.zeros(2),
    # )

    # X_res_pred = np.stack([ctrl_res.l4a_solver.get(j, "x") for j in range(N + 1)], axis=0)
    # U_res = np.stack([ctrl_res.l4a_solver.get(j, "u") for j in range(N)], axis=0)

    # # === 6) Compare under actual dynamics ===
    # X_act_from_nomU = integrate_open_loop(dyn_act, X_pred[0], U_pred, dt, lambda x, u: x, method="rk4")
    # X_act_from_resU = integrate_open_loop(dyn_act, X_res_pred[0], U_res, dt, lambda x, u: x, method="rk4")
    # fig2, _ = plot_three(X_pred, X_act_from_nomU, X_act_from_resU,
    #                      names=("vx", "vy", "wz"),
    #                      ref_xy=refline_xy,
    #                      track_left=left_dist,
    #                      track_right=right_dist,
    #                      idx_xy=("x", "y"),)

    plt.show()
    print("[done] residual MPC validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
