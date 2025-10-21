# integrators.py
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

Array = np.ndarray
DynFun = Callable[[float, Array], Array]  # f(t, x) -> xdot

# ------------------------
# One-step integrators
# ------------------------


def euler_step(dyn: DynFun, t: float, x: Array, dt: float) -> Array:
    return x + dt * dyn(t, x)


def rk4_step(dyn: DynFun, t: float, x: Array, dt: float) -> Array:
    k1 = dyn(t, x)
    k2 = dyn(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = dyn(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = dyn(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# Convenience wrapper to match your snippet: rk4_step(dyn, x, dt)


def rk4_step_simple(dyn: DynFun, x: Array, dt: float) -> Array:
    return rk4_step(dyn, 0.0, x, dt)

# ------------------------
# Open-loop integration
# ------------------------


def integrate_open_loop(
    dyn: DynFun,
    x0: Array,
    U: Array,
    dt: float,
    write_u: Optional[Callable[[Array, Array], Array]] = None,
    method: str = "rk4",
) -> Array:
    """
    Integrate x_dot = f(t, x) forward using a sequence of controls U (K,nu),
    where each u[k] is written into the state via `write_u(x,u)` BEFORE the step.

    Returns X with shape (K+1, nx): the initial state plus K integrated steps.
    """
    if method.lower() == "rk4":
        step = rk4_step
    elif method.lower() == "euler":
        step = euler_step
    else:
        raise ValueError("method must be 'rk4' or 'euler'")

    X = [np.array(x0, dtype=float).copy()]
    t = 0.0

    if U is None:
        # No controls: just advance one step
        X.append(step(dyn, t, X[-1], dt))
        return np.stack(X, axis=0)

    U = np.asarray(U, dtype=float)
    for k in range(U.shape[0]):
        xk = X[-1].copy()
        uk = U[k].copy()

        # Write control into state (e.g., x[8]=jerk, x[9]=derDelta)
        if write_u is not None:
            xk = write_u(xk, uk)

        # Advance one step
        x_next = step(dyn, t, xk, dt)
        X.append(x_next)
        t += dt

    return np.stack(X, axis=0)
