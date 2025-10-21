# dynamics_wrappers.py
from __future__ import annotations
from ...mpc_core.utils.types import Array, DynFun
def make_dyn_nominal(dyn_fun):
    return lambda t,x: dyn_fun(t,x)
def make_dyn_actual_mu(dyn_fun_with_mu, mu_true: float):
    return lambda t,x: dyn_fun_with_mu(t,x,mu_true)
