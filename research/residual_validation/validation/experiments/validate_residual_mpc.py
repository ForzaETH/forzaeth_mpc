#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from forzaeth_mpc.sim.plotting.vis_run_simple_rollout import \
    plot_closed_loop_summary
from f110_msgs.msg import Wpnt
from lbmpc.l4acados.src.l4acados.models import ResidualModel
from research.residual_validation.mpc_core.MPC.l4a_STMPC import STMPC_L4A
from research.residual_validation.mpc_core.MPC.STMPC import STMPC
from research.residual_validation.mpc_core.residual.gp_interface import (build_multitask_gp,
                                                       gp_predictor_from_model,
                                                       load_gp_state,
                                                       save_gp_state,
                                                       train_gp_model)
from research.residual_validation.mpc_core.utils.integrators import integrate_open_loop
from research.residual_validation.validation.data.dataset_one_step import build_dataset_one_step
from research.residual_validation.validation.viz.plotting import (
    plot_gp_training_fit, plot_residual_surfaces_vy_wz,
    plot_residual_surfaces_vy_wz_colored_by_vx, plot_three)
from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from forzaeth_mpc.utils.indices import StateIndex


def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    return np.array([
        (wp.x_m, wp.y_m, wp.vx_mps, wp.kappa_radpm, wp.s_m, wp.d_left, wp.d_right)
        for wp in waypoints
    ], dtype=float)


def dyn_from_ctrl(ctrl: STMPC, mu_nom: float, mu_real: float, bias=(0.0, 0.0, 0.05)):
    import casadi as ca
    x = ctrl.model.x
    u = ctrl.model.u
    p = ctrl.model.p
    f = ctrl.model.f_expl_expr
    f_fun = ca.Function("f_fun", [x, u, p], [f])
    pvec = np.array(ctrl.model_params.p, dtype=float).reshape(-1)

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    b = np.array(bias, dtype=float)

    def dyn_nom(t, xv, uv):
        pvec[12] = float(mu_nom)
        return np.array(f_fun(xv, uv, pvec)).reshape(-1)

    def dyn_act(t, xv, uv):
        pvec[12] = float(mu_real)
        dx = np.array(f_fun(xv, uv, pvec)).reshape(-1)
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


def integrate_one_step(dyn, x_plant: np.ndarray, dt: float, method: str = "rk4") -> np.ndarray:
    """
    Integrate one step of length dt for an augmented plant state:
      x_plant = [ x(0:nx), u0, u1 ], with nx = len(x_plant) - 2.
    `dyn(t, x, u)` returns xdot for the nx model states. Inputs u are held constant.
    """
    x_plant = np.asarray(x_plant, dtype=float)
    if x_plant.ndim != 1 or x_plant.size < 3:
        raise ValueError("x_plant must be 1D and contain at least [x..., u0, u1].")

    nx = x_plant.size - 2
    x = x_plant[:nx]
    u = x_plant[nx:]

    def f(xx):
        # xx is a concatenated vector; we only integrate the state part
        x_local = xx[:nx]
        return np.asarray(dyn(0.0, x_local, u), dtype=float).reshape(-1)

    if method.lower() == "rk4":
        k1 = f(np.r_[x, u])
        k2 = f(np.r_[x + 0.5 * dt * k1, u])
        k3 = f(np.r_[x + 0.5 * dt * k2, u])
        k4 = f(np.r_[x + dt * k3, u])
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        # forward Euler
        x_next = x + dt * f(np.r_[x, u])

    # Reattach unchanged inputs
    x_plant_next = x_plant.copy()
    x_plant_next[:nx] = x_next
    return x_plant_next


class L4aJacAxisAdapter(ResidualModel):
    """
    Adapter to ensure residual_jac has shape (nw, N, ny) for l4acados.
    If the wrapped model already produces (nw, N, ny), we pass it through.
    If it produces (N, nw, ny), we transpose to (nw, N, ny).
    """

    def __init__(self, inner: ResidualModel):
        self.inner = inner
        self.nr = getattr(inner, "nr", None)
        self.ny = getattr(inner, "ny", None)

    def value_and_jacobian(self, y_hat_all):
        """
        Returns:
          residual_fun: (N, nw)
          residual_jac: (nw, N, ny)  <-- as required by upstream l4acados
        """
        r, J = self.inner.value_and_jacobian(y_hat_all)

        # r should be (N, nw). We leave it as-is.
        N = r.shape[0]
        nw = r.shape[1]

        # Determine ny from provided J
        if J.ndim != 3:
            raise RuntimeError(f"Expected 3D jacobian, got shape {J.shape}")

        if J.shape == (nw, N, J.shape[2]):
            # Already (nw, N, ny): pass through
            return r, J

        if J.shape == (N, nw, J.shape[2]):
            # Common alternative: (N, nw, ny) -> transpose to (nw, N, ny)
            return r, np.transpose(J, (1, 0, 2))

        # Last resort: be explicit about the expected shapes
        raise RuntimeError(
            f"Unexpected jacobian shape {J.shape}; expected (N, nw, ny) or (nw, N, ny)."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gp_iters", type=int, default=600)
    ap.add_argument("--gp_lr", type=float, default=0.05)
    ap.add_argument("--bias_vx", type=float, default=0.0)
    ap.add_argument("--bias_vy", type=float, default=0.0)
    ap.add_argument("--bias_wz", type=float, default=0.0)
    ap.add_argument("--track", default="THETRACK")
    ap.add_argument("--racecar_version", default="NUC5")
    ap.add_argument("--floor", default="dubi_hoons")
    ap.add_argument("--lengthscale", default=2.9)
    ap.add_argument("--outputscale", default=0.000155)
    ap.add_argument("--gp_noise", default=1e-4)
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
    mu_real = 0.85
    dt = 1.0 / float(stmpc_cfg.MPC_freq)
    pose_frenet = np.array([10.0, 0.0, 0.0])
    init_vel = 4.0
    cl_iter = 40

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

    # Create the two integration functions
    dyn_nom, dyn_act = dyn_from_ctrl(
        ctrl_nom, mu_nom=float(mu_nom), mu_real=float(mu_real), bias=(
            args.bias_vx, args.bias_vy, args.bias_wz))

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

    start_time = time.time()
    for k in range(cl_iter):
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
        x_plant = integrate_one_step(dyn_nom, x_plant, dt)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        elapsed_time = time.time() - start_time
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")
        start_time = time.time()

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
    # fig = plot_closed_loop_summary(
    #     refline_xy=refline_xy,
    #     left_dist=left_dist,
    #     right_dist=right_dist,
    #     x_hist=X_aux,
    #     u_hist=U_aux
    # )
    # plt.show()

    # === 2) Integrate nominal vs actual ===
    X_nom = integrate_open_loop_fxu(dyn_nom, X_pred[0], U_pred, dt, method="euler")

    # # Visualize X_norm inegrated states
    # fig = plot_closed_loop_summary(
    #     refline_xy=refline_xy,
    #     left_dist=left_dist,
    #     right_dist=right_dist,
    #     x_hist=X_nom,
    #     u_hist=U_aux
    # )
    # plt.show()

    X_act = integrate_open_loop_fxu(dyn_act, X_pred[0], U_pred, dt, method="euler")

    # # Visualize X_act integrated states
    # fig = plot_closed_loop_summary(
    #     refline_xy=refline_xy,
    #     left_dist=left_dist,
    #     right_dist=right_dist,
    #     x_hist=X_act,
    #     u_hist=U_aux
    # )
    # plt.show()

    fig1, _ = plot_three(X_aux, X_nom, X_act,
                         names=("vx", "vy", "wz"),
                         ref_xy=refline_xy,
                         track_left=left_dist,
                         track_right=right_dist,
                         idx_s=StateIndex.POS_ON_CENTER_LINE_S,
                         idx_n=StateIndex.MIN_DIST_TO_CENTER_LINE_N,
                         )

    plt.show()

    # === 3) Dataset + 4) Train GP ===
    def nominal_step(x, u, DT): return integrate_open_loop_fxu(dyn_nom, x, u[None, :], DT, method="euler")[-1]
    def actual_step(x, u, DT): return integrate_open_loop_fxu(dyn_act, x, u[None, :], DT, method="euler")[-1]
    Z_full, Y_full = build_dataset_one_step(nominal_step, actual_step, X_pred, U_pred, dt)

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    state3 = [idx_vx, idx_vy, idx_wz]

    def select_three(mat, cols):
        # If already 3-wide, assume it's [vx, vy, wz] in that order — return as-is.
        # Otherwise slice the requested columns.
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {mat.shape}")
        if mat.shape[1] == 3:
            return mat
        # sanity: ensure indices exist
        if max(cols) >= mat.shape[1]:
            raise IndexError(f"Requested columns {cols} but matrix has {mat.shape[1]} columns")
        return mat[:, cols]

    Z = select_three(Z_full, state3)  # inputs to GP (3 features)
    Y = select_three(Y_full, state3)  # targets for GP (3 outputs)

    load_gp_model_from_state_dict = getattr(args, "load_gp", False)
    state_dict_path_gp_model = getattr(args, "gp_model_path", "gp_model_state_dict.pth")
    state_dict_path_likelihood = getattr(args, "gp_lik_path", "gp_model_likelihood_state_dict.pth")
    train_data_path = getattr(args, "gp_train_data_path", "gp_model_train_data.pth")  # optional

    device = args.device if hasattr(args, "device") else "cpu"
    training_iterations = getattr(args, "gp_iters", 600)
    rng_seed = getattr(args, "gp_seed", 456)
    lr = getattr(args, "gp_lr", 0.05)

    # gp_model, likelihood = build_multitask_gp(Z, Y, device=device)
    models, likelihoods = build_multitask_gp(Z, Y, device=device)

    # # Set gp params
    # gp_model.covar_module.data_covar_module.base_kernel.lengthscale = torch.tensor(args.lengthscale)
    # gp_model.covar_module.data_covar_module.outputscale = torch.tensor(args.outputscale)
    # likelihood.noise = torch.tensor(args.gp_noise)

    # # Fix parameters (not hyperparameter opt)
    # gp_model.covar_module.data_covar_module.base_kernel.raw_lengthscale.requires_grad = False
    # gp_model.covar_module.data_covar_module.raw_outputscale.requires_grad = False
    # likelihood.raw_noise.requires_grad = False

    # if load_gp_model_from_state_dict:
    #     load_gp_state(gp_model, likelihood, state_dict_path_gp_model, state_dict_path_likelihood)
    # else:
    # gp_model, likelihood = train_gp_model(
    #     gp_model, likelihood, torch_seed=rng_seed,
    #     training_iterations=training_iterations, lr=lr, verbose=True
    # )
    models, likelihoods = train_gp_model(models, likelihoods,  # torch_seed=rng_seed,
                                         training_iterations=training_iterations, lr=lr)

    # print("\n[GP] Learned hyperparameters:")

    # lengthscale = gp_model.covar_module.data_covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    # outputscale = gp_model.covar_module.data_covar_module.outputscale.detach().cpu().numpy()
    # noise = likelihood.noise.detach().cpu().numpy()

    # print(f"  Lengthscale : {lengthscale}")
    # print(f"  Outputscale : {outputscale}")
    # print(f"  Noise       : {noise}")

    # produce a numpy-based callable like before
    # predict_mean = gp_predictor_from_model(gp_model, likelihood, device=device)
    predict_mean = gp_predictor_from_model(models, likelihoods)

    # Use the local builder that returns (nr, ·) jacobians matching L4acados expectations.
    # === Make a residual model that only touches vx, vy, wz ===
    nx = ctrl_nom.model.n_x
    ny = nx + ctrl_nom.model.n_u
    # L4acados typically uses the state as regressor; if that’s your case:
    idx_map = {
        'state': {'vx': idx_vx, 'vy': idx_vy, 'wz': idx_wz},
        'regressor': {'vx': idx_vx, 'vy': idx_vy, 'wz': idx_wz},
    }

    # Build residual model from the GP mean, then wrap it
    residual_model_base = STMPC_L4A.make_gp_residual_model(predict_mean, idx_map, nx=nx, ny=ny)
    residual_model = L4aJacAxisAdapter(residual_model_base)

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

    x_hist = []
    u_hist = []
    status_hist = []
    x_pred_seq_hist = []
    x_plant = np.array([
        pose_frenet[0], pose_frenet[1], pose_frenet[2],
        init_vel, 0.0, 0.0, 0.0, 0.0,   # vx, vy, delta, yaw_rate, accel
        0.0, 0.0                   # jerk, derDelta (helpers)
    ], dtype=float)

    start_time = time.time()
    for k in range(cl_iter):
        x0_solver = x_plant[:ctrl_res.model.n_x].copy()
        ctrl_res.l4a_solver.set(0, "lbx", x0_solver)
        ctrl_res.l4a_solver.set(0, "ubx", x0_solver)

        for i in range(ctrl_res.stmpc_config.N + 1):

            idx = np.abs(waypoints[:, 4] - ctrl_res.mpc_sd[i, 0] % ctrl_res.spline.track_length).argmin()
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
            ctrl_res.l4a_solver.set(i, "p", online_parameters)

        # 2) Solve MPC
        status = ctrl_res.l4a_solver.solve()
        status_hist.append(status)

        if not (status == 0 or status == 2):  # 0 is sucess 2 is max iter, necesary for SQP
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
            ctrl_res.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        # 3) Read optimal control (stage 0): u = [jerk, ddelta]
        u0 = ctrl_res.l4a_solver.get(0, "u").astype(float)
        u_hist.append(u0.copy())

        # Collect full horizon state predictions
        H = ctrl_res.stmpc_config.N
        nx = ctrl_res.model.n_x
        x_pred_full = np.zeros((H + 1, nx))
        for i in range(H + 1):
            x_pred_full[i, :] = ctrl_res.l4a_solver.get(i, "x")
        x_pred_seq_hist.append(x_pred_full)

        ctrl_res.mpc_sd = np.array([
            ctrl_res.l4a_solver.get(j, "x")[:2] for j in range(ctrl_res.stmpc_config.N + 1)
        ])

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        x_plant = integrate_one_step(dyn_act, x_plant, dt)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        elapsed_time = time.time() - start_time
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")
        start_time = time.time()

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
    nx = ctrl_res.model.n_x
    nu = ctrl_res.model.n_u
    X_pred = np.stack([ctrl_res.l4a_solver.get(j, "x") for j in range(N + 1)], axis=0)
    U_pred = np.stack([ctrl_res.l4a_solver.get(j, "u") for j in range(N)], axis=0)

    # Visualize the last prediction horizon states and inputs
    U_aux = np.vstack([U_pred, U_pred[-1:]])
    X_aux = np.asarray(X_pred, dtype=float)
    U_aux = np.asarray(U_aux, dtype=float)
    fig = plot_closed_loop_summary(
        refline_xy=refline_xy,
        left_dist=left_dist,
        right_dist=right_dist,
        x_hist=X_aux,
        u_hist=U_aux
    )
    plt.show()

    # === 2) Integrate nominal vs actual ===
    X_nom = integrate_open_loop_fxu(dyn_nom, X_pred[0], U_pred, dt, method="euler")
    X_act = integrate_open_loop_fxu(dyn_act, X_pred[0], U_pred, dt, method="euler")
    fig1, _ = plot_three(X_aux, X_nom, X_act,
                         names=("vx", "vy", "wz"),
                         ref_xy=refline_xy,
                         track_left=left_dist,
                         track_right=right_dist,
                         idx_s=StateIndex.POS_ON_CENTER_LINE_S,   # <-- Frenet longitudinal coordinate
                         idx_n=StateIndex.MIN_DIST_TO_CENTER_LINE_N,   # <-- Frenet lateral coordinate
                         )
    plt.show()

    plot_gp_training_fit(Z, Y, models, likelihoods, device=device, titles=("vx", "vy", "wz"))
    plt.show()

    plot_residual_surfaces_vy_wz(Z, Y)
    plt.show()

    plot_residual_surfaces_vy_wz_colored_by_vx(Z, Y)
    plt.show()

    print("[done] residual MPC validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
