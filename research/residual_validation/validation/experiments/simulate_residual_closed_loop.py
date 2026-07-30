#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from forzaeth_mpc.sim.plotting.vis_run_simple_rollout import \
    plot_closed_loop_summary
from f110_msgs.msg import Wpnt
from l4acados.models import ResidualModel
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
    plot_dict_kept_dropped, plot_gp_training_fit, plot_residual_surfaces_vy_wz,
    plot_residual_surfaces_vy_wz_colored_by_vx, plot_three)
from forzaeth_mpc.config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig, load_car_config_ros,
                        load_pacejka_tire_config_ros, load_STMPC_config_ros,
                        load_trailing_config_ros)
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from forzaeth_mpc.utils.indices import StateIndex
from sklearn.neighbors import KDTree


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


def integrate_one_step(dyn,
                       x_plant: np.ndarray,
                       dt: float,
                       method: str = "rk4",
                       noise_vars: Optional[Sequence[float]] = None,
                       noise_indices: Optional[Sequence[int]] = None,
                       rng: Optional[np.random.Generator] = None,
                       ) -> np.ndarray:
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

    # Add gaussian noise
    if noise_vars is not None and noise_indices is not None:
        if rng is None:
            rng = np.random.default_rng()
        for idx, var in zip(noise_indices, noise_vars):
            x_next[idx] += rng.normal(0.0, np.sqrt(var))

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


def lap_times_from_s(s_series: np.ndarray, dt: float, track_length: float) -> List[float]:
    """
    Compute per-lap times by unwrapping s and detecting crossings of s0 + k*track_length.
    Assumes uniform sampling with step dt.
    Returns a list of lap durations [t_lap1, t_lap2, ...] for all fully completed laps.
    """
    s_series = np.asarray(s_series, dtype=float)
    if s_series.ndim != 1 or len(s_series) < 2:
        return []

    # Unwrap s after the cut
    unwrapped = s_series.copy()
    offset = 0.0
    for i in range(1, len(unwrapped)):
        ds = s_series[i] - s_series[i - 1]
        if ds < -0.5 * track_length:
            offset += track_length
        unwrapped[i] += offset

    s0 = unwrapped[0]
    t = np.arange(len(unwrapped), dtype=float) * dt

    total_prog = unwrapped[-1] - s0
    n_laps = int(np.floor(total_prog / track_length))
    if n_laps <= 0:
        return []

    # Find lap-crossing times
    cross_times = []
    for k in range(1, n_laps + 1):
        thresh = s0 + k * track_length
        idx = np.searchsorted(unwrapped, thresh)
        if 0 < idx < len(unwrapped):
            s_prev, s_curr = unwrapped[idx - 1], unwrapped[idx]
            t_prev, t_curr = t[idx - 1], t[idx]
            if s_curr != s_prev:
                frac = (thresh - s_prev) / (s_curr - s_prev)
                cross_times.append(t_prev + frac * (t_curr - t_prev))

    if not cross_times:
        return []

    # Lap durations relative to the first start-line crossing
    lap_times = [cross_times[0]]
    for i in range(1, len(cross_times)):
        lap_times.append(cross_times[i] - cross_times[i - 1])

    return lap_times


def compute_lateral_deviation(x_hist: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of the lateral deviation [m]
    from the Frenet-state trajectories.

    Assumes the lateral deviation 'n' is the 2nd state (index 1).
    """
    if x_hist.ndim != 2 or x_hist.shape[1] < 2:
        raise ValueError("x_hist must be (N, nx) and contain the lateral deviation at index 1.")

    lateral_dev = x_hist[:, 1]
    mean_dev = float(np.mean(np.abs(lateral_dev)))
    std_dev = float(np.std(lateral_dev))
    return mean_dev, std_dev


def get_kernel_params(model, likelihood):
    # gpytorch: RBF with ARD + ScaleKernel
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)  # (nz,)
    sf2 = model.covar_module.outputscale.detach().cpu().item()                          # scalar
    sn2 = likelihood.noise.detach().cpu().item()                                        # scalar
    return {"lengthscale": ls, "outputscale": sf2, "noise": sn2}


class FitcRuntime:
    """
    Minimal FITC precompute + mean/Jacobian evaluator for one output head.
    Uses RBF-ARD kernel; hyperparams fixed from trained GP.
    """

    def __init__(self, Z_dict: np.ndarray, y_dict: np.ndarray, params: dict):
        self.Z = Z_dict      # (M, nz)
        self.y = y_dict      # (M,)
        self.ls = params["lengthscale"]  # (nz,)
        self.sf2 = params["outputscale"]  # scalar
        self.sn2 = params["noise"]       # scalar
        self._pre = None

    def _rbf(self, A, B):
        # A:(p,nz), B:(q,nz) -> (p,q)
        D = (A[:, None, :] - B[None, :, :]) / self.ls
        return self.sf2 * np.exp(-np.sum(D * D, axis=2))

    def _rbf_dx(self, x, B):
        # gradient wrt x of k(x, B_j)
        # x:(nz,), B:(q,nz) -> (q,nz)
        d = (x[None, :] - B) / (self.ls**2)
        kxB = self._rbf(x[None, :], B).reshape(-1, 1)  # (q,1)
        return -2.0 * kxB * d                          # (q,nz)

    def precompute_tick(self, U: np.ndarray):
        """
        U: inducing inputs (m̃, nz)
        Precompute Λ^{-1}, K_UU, chol(S) where S = K_UU + K_UZ Λ^{-1} K_ZU (Woodbury).
        """
        Z, y = self.Z, self.y                   # (M, nz), (M,)

        # Kernel blocks
        # If you have a fast diagonal routine, prefer it to forming K_ZZ fully.
        K_ZU = self._rbf(Z, U)                  # (M, m̃)
        K_UU = self._rbf(U, U)
        # Scaled jitter helps conditioning more predictably than a bare 1e-9
        K_UU = K_UU + (1e-6 * getattr(self, "sf2", 1.0)) * np.eye(K_UU.shape[0])

        # Diagonal of K_ZZ; this is just k(z_i, z_i). If your RBF amplitude is sf2, this is sf2 * np.ones(M).
        K_ZZ_diag = np.full(Z.shape[0], self.sf2, dtype=float)

        # Compute diag(Q_ZZ) = diag(K_ZU K_UU^{-1} K_UZ)
        # Do NOT invert; use solves with Cholesky.
        K_UU_chol = np.linalg.cholesky(K_UU)
        # Solve K_UU^{-1} K_UZ = (K_UU^{-1} K_ZU^T)^T in two triangular solves
        tmp = np.linalg.solve(K_UU_chol, K_ZU.T)           # (m̃, M)
        KUU_inv_KUZ = np.linalg.solve(K_UU_chol.T, tmp)    # (m̃, M)
        # diag(K_ZU K_UU^{-1} K_UZ) = sum over columns (elementwise) of K_ZU * (K_UU^{-1} K_UZ)^T
        Q_ZZ_diag = np.sum(K_ZU * KUU_inv_KUZ.T, axis=1)   # (M,)

        # Λ as a vector: λ_i = k(z_i,z_i) - q_ii + σ_n^2
        lam = K_ZZ_diag - Q_ZZ_diag + self.sn2             # (M,)
        lam = np.maximum(lam, 1e-12 * self.sf2)
        inv_lam = 1.0 / lam

        # Woodbury middle term: S = K_UU + K_UZ Λ^{-1} K_ZU
        # A = K_ZU^T Λ^{-1}  (but we never form Λ^{-1}; we right-multiply by inv_lam elementwise)
        A = (K_ZU.T * inv_lam[None, :])                    # (m̃, M)
        S = K_UU + A @ K_ZU                                 # (m̃, m̃)
        S_chol = np.linalg.cholesky(S)

        # alpha = Λ^{-1} y - Λ^{-1} K_ZU S^{-1} K_UZ Λ^{-1} y
        rhs = A @ y                                        # (m̃,)
        v = np.linalg.solve(S_chol, rhs)
        w = np.linalg.solve(S_chol.T, v)                   # S^{-1} K_UZ Λ^{-1} y
        alpha = inv_lam * (y - K_ZU @ w)                   # (M,)

        # Cache everything needed for fast mean/Jac evaluation
        self._pre = {
            "U": U,
            "inv_lam": inv_lam,    # vector, not matrix
            "K_ZU": K_ZU,
            "K_UU_chol": K_UU_chol,
            "S_chol": S_chol,
            "alpha": alpha,
            "w": w,
        }

    def mean_and_jac(self, x: np.ndarray):
        """
        x:(nz,) -> mean scalar, jacobian wrt x:(nz,)
        μ̃(x) = Q_{xZ} (Q_{ZZ}+Λ)^{-1} y
              = K_{xU} K_UU^{-1} K_{UZ} (Q_{ZZ}+Λ)^{-1} y
        Using precomputed Woodbury, this reduces to:
        μ̃(x) = [K_{xZ} - K_{xU}(S^{-1}K_UZΛ^{-1})]^T Λ^{-1} y
              = k_xZ^T * alpha  -  k_xU^T * (S^{-1}K_UZΛ^{-1} y)
        """
        U, inv_lam, K_ZU, S_chol, alpha, y, w = \
            self._pre["U"], self._pre["inv_lam"], self._pre["K_ZU"], self._pre["S_chol"], self._pre["alpha"], self.y, self._pre["w"]

        k_xZ = self._rbf(x[None, :], self.Z).reshape(-1)     # (M,)
        k_xU = self._rbf(x[None, :], U).reshape(-1)          # (m̃,)

        mu = float(k_xZ @ alpha - k_xU @ w)

        # Jacobian via kernel derivatives
        dkxZ_dx = self._rbf_dx(x, self.Z)                    # (M,nz)
        dkxU_dx = self._rbf_dx(x, U)                         # (m̃,nz)
        dmu = dkxZ_dx.T @ alpha - dkxU_dx.T @ w              # (nz,)

        return mu, dmu


class FitcResidual3Only(ResidualModel):
    def __init__(self, Z_dict, Y_dict, kern_params, idx_map, nx, ny, m_ind=10, place="equal"):
        # split dictionary per head (vx,vy,wz)
        self.Zd = Z_dict                        # (M,3)  you already selected_three(..)
        self.Yd = [Y_dict[:, i] for i in range(3)]
        self.params = kern_params               # list of 3 dicts
        self.idx_vx = idx_map['regressor']['vx']
        self.idx_vy = idx_map['regressor']['vy']
        self.idx_wz = idx_map['regressor']['wz']
        self.nx, self.ny = nx, ny
        self.m_ind = m_ind
        self.place = place
        # initialized at first call
        self.fitc_runtimes = [None, None, None]

        # Create kdtree
        ls_vx = self.params[0]["lengthscale"]
        ls_vy = self.params[1]["lengthscale"]
        ls_wz = self.params[2]["lengthscale"]
        ls3 = np.power(ls_vx * ls_vy * ls_wz, 1 / 3)   # shape (3,)
        self._ls3 = ls3
        self._Zd_scaled = self.Zd / self._ls3
        self._tree = KDTree(self._Zd_scaled)

    def _regressor3(self, y_full_row):
        # y_full_row: (ny,)
        return np.array([y_full_row[self.idx_vx],
                         y_full_row[self.idx_vy],
                         y_full_row[self.idx_wz]], dtype=float)

    def _choose_inducing(self, Y_hat_all_3d):
        N = len(Y_hat_all_3d)
        R = min(4, N)
        rep_idx = np.linspace(0, N - 1, R).astype(int)
        reps = Y_hat_all_3d[rep_idx, :]

        k_total = self.m_ind
        # ask slightly more neighbors to combat dupes; we'll prune later
        k_per = max(1, int(np.ceil(1.5 * k_total / R)))

        cand = []
        for r in reps:
            rs = (r / self._ls3).reshape(1, -1)
            _, idx = self._tree.query(rs, k=k_per)
            cand.extend(idx[0].tolist())

        # de-duplicate while preserving order
        seen = set()
        idxs = []
        for i in cand:
            if i not in seen:
                idxs.append(i)
                seen.add(i)
            if len(idxs) == k_total:
                break

        # if still short (rare), top up with farthest-point among remaining
        if len(idxs) < k_total:
            pool = np.array([i for i in range(self._Zd_scaled.shape[0]) if i not in seen])
            if pool.size > 0:
                Usel = self._Zd_scaled[idxs, :]
                remain = self._Zd_scaled[pool, :]
                # farthest from current selection
                d2 = np.min(np.sum((remain[:, None, :] - Usel[None, :, :])**2, axis=2), axis=1)
                order = np.argsort(-d2)
                take = pool[order[:(k_total - len(idxs))]].tolist()
                idxs.extend(take)

        U = self.Zd[idxs[:k_total], :]
        return U

    def value_and_jacobian(self, y_hat_all):
        # y_hat_all: (N, ny)
        N = y_hat_all.shape[0]
        Y3 = np.stack([self._regressor3(y_hat_all[i]) for i in range(N)], axis=0)  # (N,3)

        # 1) Pick inducing along the linearization trajectory (core FITC idea)
        U = self._choose_inducing(Y3)    # (m̃,3)

        # 2) Precompute Woodbury for each output head this tick
        if self.fitc_runtimes[0] is None:
            self.fitc_runtimes = [FitcRuntime(self.Zd, self.Yd[h], self.params[h]) for h in range(3)]
        for h in range(3):
            self.fitc_runtimes[h].precompute_tick(U)

        # 3) Evaluate mean & Jacobian wrt regressor for all N stages
        r = np.zeros((N, 3))
        J = np.zeros((N, 3, y_hat_all.shape[1]))   # we'll expand to full ny later

        for i in range(N):
            z3 = Y3[i]
            for h in range(3):
                mu, dmu_dz3 = self.fitc_runtimes[h].mean_and_jac(z3)
                r[i, h] = mu
                J[i, h, self.idx_vx] = dmu_dz3[0]
                J[i, h, self.idx_vy] = dmu_dz3[1]
                J[i, h, self.idx_wz] = dmu_dz3[2]

        # L4acados expects (N, nw) and (nw, N, ny) — let adapter transpose
        return r, J


# === dictionary settings ===
USE_LENGTHSCALES = True     # scale features before distances? (recommended)
M_MAX = 40                 # max points to keep
# If you trained an ARD RBF, pass its lengthscales here (shape: (nz,))
LENGTHSCALES = np.asarray([1.0, 1.0, 1.0])  # replace with your ARD ls if available
MIN_DIST = 0.15


def update_dictionary(Z_list, Y_list, z_new, y_new,
                      Z_vis_kept, Z_vis_dropped,
                      idx3,
                      m_max=M_MAX,
                      ls3=LENGTHSCALES,   # must be shape (3,)
                      min_dist=MIN_DIST):
    """
    Selectivity in 3D GP-regressor space, store full z/y.
    """
    z_new = np.asarray(z_new, dtype=float).reshape(-1)
    y_new = np.asarray(y_new, dtype=float).reshape(-1)
    idx3 = np.asarray(idx3, dtype=int).reshape(-1)
    if idx3.size != 3:
        raise ValueError(f"idx3 must have 3 indices, got {idx3}")
    if ls3 is not None:
        ls3 = np.asarray(ls3, dtype=float).reshape(-1)
        if ls3.size != 3:
            raise ValueError(f"ls3 must have shape (3,), got {ls3.shape}")

    # Helper: scale 3D vector(s)
    def _scale3(A3):
        return A3 / ls3 if ls3 is not None else A3

    # ---- empty dictionary: just add ----
    if len(Z_list) == 0:
        Z_list.append(z_new.copy())
        Y_list.append(y_new.copy())
        Z_vis_kept.append(z_new.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # ---- stack existing ----
    Z = np.vstack(Z_list)  # (M, nz)
    # project to 3D GP subspace
    Z3 = Z[:, idx3]       # (M, 3)
    z3n = z_new[idx3]      # (3,)

    # scale
    Z3s = _scale3(Z3)
    z3ns = _scale3(z3n)[None, :]  # (1,3)

    # ---- NN distance to dictionary (in scaled 3D) ----
    d2 = np.sum((Z3s - z3ns)**2, axis=1)  # (M,)
    dmin = float(np.sqrt(np.min(d2)))

    # ---- too close → drop ----
    if dmin < min_dist:
        Z_vis_dropped.append(z_new.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # ---- capacity available → append ----
    if len(Z_list) < m_max:
        Z_list.append(z_new.copy())
        Y_list.append(y_new.copy())
        Z_vis_kept.append(z_new.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped

    # ---- full: add candidate, then remove least-informative ----
    Zcand = np.vstack([Z, z_new[None, :]])           # (M+1, nz)
    Ycand = np.vstack([np.vstack(Y_list), y_new[None, :]])  # (M+1, ny) if you need Y in decisions later
    Z3cs = _scale3(Zcand[:, idx3])                  # (M+1, 3)

    # Compute per-point NN distance among candidates (O((M+1)^2) ok for moderate M)
    G = Z3cs @ Z3cs.T
    nrm = np.sum(Z3cs**2, axis=1, keepdims=True)
    D2 = nrm + nrm.T - 2.0 * G
    np.fill_diagonal(D2, np.inf)
    nn = np.sqrt(np.min(D2, axis=1))                 # (M+1,)

    drop_idx = int(np.argmin(nn))                     # smallest NN distance → most redundant

    if drop_idx == len(Z_list):
        # New point is least informative → drop it
        Z_vis_dropped.append(z_new.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped
    else:
        # Replace existing point with the new one
        Z_vis_dropped.append(Z_list[drop_idx].copy())  # record which one we removed
        Z_list[drop_idx] = z_new.copy()
        Y_list[drop_idx] = y_new.copy()
        Z_vis_kept.append(z_new.copy())
        return Z_list, Y_list, Z_vis_kept, Z_vis_dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gp_iters", type=int, default=300)
    ap.add_argument("--gp_lr", type=float, default=0.05)
    ap.add_argument("--bias_vx", type=float, default=1.0)
    ap.add_argument("--bias_vy", type=float, default=0.0)
    ap.add_argument("--bias_wz", type=float, default=0.0)
    ap.add_argument("--track", default="THETRACK")
    ap.add_argument("--racecar_version", default="NUC5")
    ap.add_argument("--floor", default="dubi_hoons")
    ap.add_argument("--lengthscale", default=2.9)
    ap.add_argument("--outputscale", default=0.000155)
    ap.add_argument("--gp_noise", default=1e-4)
    ap.add_argument("--noise_vx", default=1e-5)
    ap.add_argument("--noise_vy", default=1e-5)
    ap.add_argument("--noise_wz", default=1e-5)
    ap.add_argument("--fitc_m", type=int, default=20, help="Number of inducing points along the horizon")
    ap.add_argument("--fitc_place", choices=["equal", "kmeans"], default="equal",
                    help="How to place inducing points on the predicted trajectory")

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
    mu_real = 0.8
    dt = 1.0 / float(stmpc_cfg.MPC_freq)
    pose_frenet = np.array([15.0, 0.0, 0.0])
    init_vel = 4.0
    cl_iter = 650
    use_fitc = False

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

    idx_vx = StateIndex.VELOCITY_V_X.value
    idx_vy = StateIndex.VELOCITY_V_Y.value
    idx_wz = StateIndex.YAW_RATE.value
    state3 = [idx_vx, idx_vy, idx_wz]

    sigma_vx = getattr(args, "noise_vx", 0.01)
    sigma_vy = getattr(args, "noise_vy", 0.01)
    sigma_wz = getattr(args, "noise_wz", 0.01)
    noise_vars = [sigma_vx, sigma_vy, sigma_wz]

    def select_three(mat: np.ndarray, cols):
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {mat.shape}")
        if mat.shape[1] == 3:
            return mat
        if max(cols) >= mat.shape[1]:
            raise IndexError(f"Requested columns {cols} but matrix has {mat.shape[1]} columns")
        return mat[:, cols]

    # Lists of FULL (nx-wide) samples; we'll slice to 3 cols after stacking.
    Z_full_list, Y_full_list = [], []
    Z_vis_kept = []
    Z_vis_dropped = []

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

        # >>> GP: collect features and the first predicted state (k+1|k)
        # Do this RIGHT AFTER you have u0 and x_pred_full
        nx = ctrl_nom.model.n_x
        x_pred_1 = x_pred_full[1, :].astype(float)
        z_k_full = x0_solver[:nx].astype(float)

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        x_plant = integrate_one_step(dyn_act, x_plant, dt, noise_vars=noise_vars, noise_indices=state3)

        # if k < 50:
        x_plant_model = x_plant[:nx].astype(float)
        y_k_full = (x_plant_model - x_pred_1)                # (ny,)
        z_new = z_k_full.reshape(-1)                         # (nz,)
        y_new = y_k_full.reshape(-1)
        Z_full_list, Y_full_list, Z_vis_kept, Z_vis_dropped = update_dictionary(
            Z_full_list, Y_full_list, z_new, y_new, Z_vis_kept, Z_vis_dropped, state3)

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

    plot_dict_kept_dropped(
        Z_vis_kept,
        Z_vis_dropped,
        dims=(
            idx_vx,
            idx_vy,
            idx_wz),
        title="Dictionary: kept vs dropped")
    plt.show()

    x_hist_nom = x_hist

    # === GP TRAINING: post-loop ===
    Z_full = np.vstack(Z_full_list).astype(np.float64)   # (N, nx)
    Y_full = np.vstack(Y_full_list).astype(np.float64)   # (N, nx)
    Z = select_three(Z_full, state3)                     # (N, 3) -> [vx, vy, wz]
    Y = select_three(Y_full, state3)
    print(f"Building GP")

    device = args.device if hasattr(args, "device") else "cpu"
    training_iterations = getattr(args, "gp_iters", 600)
    lr = getattr(args, "gp_lr", 0.05)

    assert Z.ndim == 2 and Y.ndim == 2 and Z.shape[1] == 3 and Y.shape[1] == 3, \
        f"Expect (N,3); got Z{Z.shape}, Y{Y.shape}"
    assert Z.shape[0] == Y.shape[0] and Z.shape[0] > 0, "Empty or mismatched Z/Y."

    models, likelihoods = build_multitask_gp(Z, Y, device=device)

    # for model in models:
    #     model.covar_module.data_covar_module.base_kernel.lengthscale = torch.tensor(args.lengthscale)
    #     model.covar_module.data_covar_module.outputscale = torch.tensor(args.outputscale)

    for likelihood in likelihoods:
        likelihood.noise = torch.tensor(args.gp_noise)

    models, likelihoods = train_gp_model(
        models, likelihoods,
        training_iterations=training_iterations, lr=lr
    )
    kern_params = [get_kernel_params(m, l) for m, l in zip(models, likelihoods)]
    predict_mean = gp_predictor_from_model(models, likelihoods)
    print(f"GP Built!")

    # Use the local builder that returns (nr, ·) jacobians matching L4acados expectations.
    # === Make a residual model that only touches vx, vy, wz ===
    nx = ctrl_nom.model.n_x
    ny = nx + ctrl_nom.model.n_u
    # L4acados typically uses the state as regressor:
    idx_map = {
        'state': {'vx': idx_vx, 'vy': idx_vy, 'wz': idx_wz},
        'regressor': {'vx': idx_vx, 'vy': idx_vy, 'wz': idx_wz},
    }

    # Build residual model from the GP mean, then wrap it
    if use_fitc:
        # dictionary already in 3D space (you called select_three earlier)
        fitc_residual = FitcResidual3Only(
            Z_dict=Z,                  # (M,3)
            Y_dict=Y,                  # (M,3)
            kern_params=kern_params,   # list of 3 dicts
            idx_map=idx_map, nx=nx, ny=ny,
            m_ind=args.fitc_m, place=args.fitc_place
        )
        residual_model = L4aJacAxisAdapter(fitc_residual)
    else:
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

    for k in range(cl_iter):
        start_time = time.time()
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
        try:
            status = ctrl_res.l4a_solver.solve()
        except Exception as e:
            msg = str(e)
            m = re.search(r'status\s+(\d+)', msg)
            status = int(m.group(1)) if m else -1
            ctrl_res.apply_warm_start(pose_frenet=[x_plant[0], x_plant[1], x_plant[2]])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        status_hist.append(status)

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

        elapsed_time = time.time() - start_time

        # 4) Apply control to the "plant":
        # Put current control into the helper states (indices 8: jerk, 9: derDelta)
        x_plant[8] = u0[0]  # jerk = d(accel)/dt from MPC
        x_plant[9] = u0[1]  # derDelta = d(delta)/dt from MPC

        # One-step forward Euler integration of your nonlinear dynamics
        x_plant = integrate_one_step(dyn_act, x_plant, dt, noise_vars=noise_vars, noise_indices=state3)

        # 5) Log plant state
        x_hist.append(x_plant.copy())

        # (optional) simple timing print
        # if (k % 20) == 0:
        print(f"[{k:04d}] status={status}, u0={u0}, elapsed={elapsed_time*1e3:.1f} ms")

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

    x_hist_res = x_hist

    plot_gp_training_fit(Z, Y, models, likelihoods, device=device, titles=("vx", "vy", "wz"))
    plt.show()

    # plot_residual_surfaces_vy_wz(Z, Y)
    # plt.show()

    # plot_residual_surfaces_vy_wz_colored_by_vx(Z, Y)
    # plt.show()

    # === PERFORMANCE COMPARISON (lap times & lateral deviation) ===
    L = float(ctrl_nom.spline.track_length)   # track length
    s_nom = x_hist_nom[:, 0]                  # s is the first state
    s_res = x_hist_res[:, 0]

    lap_times_nom = lap_times_from_s(s_nom, dt, L)
    lap_times_res = lap_times_from_s(s_res, dt, L)

    mean_dev_nom, std_dev_nom = compute_lateral_deviation(x_hist_nom)
    mean_dev_res, std_dev_res = compute_lateral_deviation(x_hist_res)

    def fmt_laps(laps):
        return " / ".join(f"{t:.3f}s" for t in laps) if laps else "—"

    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Track length: {L:.2f} m")
    print(f"Nominal MPC -> laps finished: {len(lap_times_nom)} | lap times: {fmt_laps(lap_times_nom)} | "
          f"mean lateral deviation: {mean_dev_nom:.3f} ± {std_dev_nom:.3f} m")
    print(f"LB-MPC      -> laps finished: {len(lap_times_res)} | lap times: {fmt_laps(lap_times_res)} | "
          f"mean lateral deviation: {mean_dev_res:.3f} ± {std_dev_res:.3f} m")

    print("[done] residual MPC validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
