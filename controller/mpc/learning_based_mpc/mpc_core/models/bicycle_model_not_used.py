# bicycle_model.py
from __future__ import annotations
from typing import Dict, Any
import casadi as ca
def build_bicycle_model(cfg_car: Dict[str, Any], cfg_tire: Dict[str, Any]) -> Dict[str, Any]:
    nx, nu, np_ = 7, 2, 1  # μ
    x = ca.SX.sym("x", nx); u = ca.SX.sym("u", nu); p = ca.SX.sym("p", np_)
    s,n,psi,vx,vy,wz,delta = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
    jerk, ddelta = u[0], u[1]; mu = p[0]
    m = float(cfg_car.get("m", 3.0)); Iz = float(cfg_car.get("Iz", 0.2))
    lf = float(cfg_car.get("lf", 0.15)); lr = float(cfg_car.get("lr", 0.15))
    Cf = float(cfg_tire.get("Cf", 30.0)); Cr = float(cfg_tire.get("Cr", 35.0))
    Fyf = -Cf * (delta - (vy + lf*wz)/(vx + 1e-3))
    Fyr = -Cr * (-(vy - lr*wz)/(vx + 1e-3))
    Fyf = ca.fmin(ca.fmax(Fyf, -mu*m*9.81), mu*m*9.81)
    Fyr = ca.fmin(ca.fmax(Fyr, -mu*m*9.81), mu*m*9.81)
    s_dot   = vx*ca.cos(psi) - vy*ca.sin(psi)
    n_dot   = vx*ca.sin(psi) + vy*ca.cos(psi)
    psi_dot = wz
    vx_dot = jerk
    vy_dot = (Fyf + Fyr)/m - vx*wz
    wz_dot = (lf*Fyf - lr*Fyr)/Iz
    delta_dot = ddelta
    f_expl = ca.vertcat(s_dot, n_dot, psi_dot, vx_dot, vy_dot, wz_dot, delta_dot)
    return {"name":"spatial_bicycle_mu","x":x,"u":u,"p":p,"f_expl":f_expl,"nx":nx,"nu":nu,"np":np_}
