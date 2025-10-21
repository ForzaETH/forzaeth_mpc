# open_loop.py
from __future__ import annotations
from ...mpc_core.utils.integrators import integrate_open_loop
def open_loop_two(dyn_nom, dyn_act, x0, U, dt, write_u=None, method="rk4"):
    X_nom = integrate_open_loop(dyn_nom, x0, U, dt, write_u, method)
    X_act = integrate_open_loop(dyn_act, x0, U, dt, write_u, method)
    return X_nom, X_act
