from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

State = np.ndarray
Control = Optional[np.ndarray]
DynFun = Callable[[State, Control], State]


def euler_step(f: DynFun, x: State, u: Control, dt: float) -> State:
    """
    Forward Euler step: x_{k+1} = x_k + dt * f(x_k, u_k)
    f must return dx/dt with same shape as x.
    """
    dx = np.asarray(f(x, u), dtype=float)
    return x + dt * dx


def rk4_step(f: DynFun, x: State, u: Control, dt: float) -> State:
    """
    Classic Runge–Kutta 4th order step.
    f(x, u) must return dx/dt with same shape as x.
    """
    k1 = np.asarray(f(x, u), dtype=float)
    k2 = np.asarray(f(x + 0.5 * dt * k1, u), dtype=float)
    k3 = np.asarray(f(x + 0.5 * dt * k2, u), dtype=float)
    k4 = np.asarray(f(x + dt * k3, u), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_n_steps(
    f: DynFun,
    x0: State,
    u_seq: Optional[np.ndarray],
    dt: float,
    n: int,
    *,
    method: str = "rk4"
) -> np.ndarray:
    """
    Convenience integrator for a known control sequence.

    Args:
        f: dynamics f(x, u) -> dx/dt
        x0: initial state (nx,)
        u_seq: sequence (n, nu) or None (treated as zeros / not used)
        dt: time step
        n: number of steps
        method: "rk4" or "euler"

    Returns:
        x_hist: array (n+1, nx) with x[0]=x0
    """
    x = np.asarray(x0, dtype=float).copy()
    nx = x.shape[0]
    x_hist = np.zeros((n + 1, nx), dtype=float)
    x_hist[0] = x

    if u_seq is None:
        u_seq = np.zeros((n, 0), dtype=float)

    step = rk4_step if method.lower() == "rk4" else euler_step

    for k in range(n):
        u = u_seq[k] if k < len(u_seq) else None
        x = step(f, x, u, dt)
        x_hist[k + 1] = x

    return x_hist
