# residual_mpc.py
from __future__ import annotations
import numpy as np, l4acados as l4a
def solve_once_residual_mpc(l4a_solver: l4a.ResidualLearningMPC, x0: np.ndarray):
    _ = l4a_solver.solve(x0); solver = l4a_solver.internal_solver
    H = getattr(solver.options, "N_horizon", None) or solver.dims.N
    X = []; U = []
    for j in range(H+1):
        X.append(np.asarray(solver.get(j, "x"), dtype=float))
        if j < H:
            U.append(np.asarray(solver.get(j, "u"), dtype=float))
    return np.stack(X,axis=0), np.stack(U,axis=0)
