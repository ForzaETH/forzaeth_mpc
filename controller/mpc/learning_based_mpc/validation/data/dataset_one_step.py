# dataset_one_step.py
from __future__ import annotations
import numpy as np
from ...mpc_core.utils.indices import IDX, SUBSET_RESIDUAL
def build_dataset_one_step(nominal_step, actual_step, Xp, Up, dt, subset=SUBSET_RESIDUAL):
    H = Up.shape[0]; nx = Xp.shape[1]; nu = Up.shape[1]
    Z = np.zeros((H, nx+nu), dtype=float); Y = np.zeros((H, len(subset)), dtype=float)
    for k in range(H):
        xk = Xp[k]; uk = Up[k]
        xn = nominal_step(xk, uk, dt); xa = actual_step(xk, uk, dt)
        Z[k,:] = np.concatenate([xk, uk]); dx = xa - xn
        Y[k,:] = np.array([dx[IDX[name]] for name in subset], dtype=float)
    return Z, Y
