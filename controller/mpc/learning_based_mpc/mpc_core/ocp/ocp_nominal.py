# ocp_nominal.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np, casadi as ca
from acados_template import AcadosOcp, AcadosModel
def build_nominal_ocp(model_dict: Dict[str,Any], mpc_cfg: Dict[str,Any]) -> AcadosOcp:
    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = model_dict["name"]; model.x = model_dict["x"]; model.u = model_dict["u"]; model.p = model_dict["p"]
    model.f_expl_expr = model_dict["f_expl"]; ocp.model = model
    N = int(mpc_cfg.get("N", 30)); dt = float(mpc_cfg.get("dt", 0.04))
    ocp.dims.N = N; ocp.solver_options.tf = N*dt
    nx = int(model_dict["nx"]); nu = int(model_dict["nu"]); np_ = int(model_dict["np"])
    ocp.dims.nx = nx; ocp.dims.nu = nu; ocp.dims.np = np_
    Q = np.diag(mpc_cfg.get("Q_diag", [1,5,1,0.2,1,0.2,0.1])); R = np.diag(mpc_cfg.get("R_diag", [0.01,0.01]))
    x_ref = np.zeros(nx); u_ref = np.zeros(nu)
    x = model.x; u = model.u
    ocp.model.cost_y_expr = ca.vertcat(x - x_ref, u - u_ref)
    ocp.model.cost_y_expr_e = x - x_ref
    ocp.cost.cost_type = "EXTERNAL"; ocp.cost.cost_type_e = "EXTERNAL"
    ocp.cost.W = np.block([[Q, np.zeros((nx,nu))],[np.zeros((nu,nx)), R]]); ocp.cost.W_e = Q
    vx_min, vx_max = mpc_cfg.get("vx_bounds", [0.0, 12.0]); delta_max = mpc_cfg.get("delta_max", 0.6)
    ocp.constraints.lbx = np.array([ -np.inf, -2.5, -np.inf, vx_min, -3.0, -5.0, -delta_max ])
    ocp.constraints.ubx = np.array([  np.inf,  2.5,  np.inf, vx_max,  3.0,  5.0,  delta_max ])
    ocp.constraints.idxbx = np.arange(nx, dtype=int)
    jerk_max = mpc_cfg.get("jerk_max", 8.0); ddelta_max = mpc_cfg.get("ddelta_max", 6.0)
    ocp.constraints.lbu = np.array([-jerk_max, -ddelta_max]); ocp.constraints.ubu = np.array([jerk_max, ddelta_max])
    ocp.constraints.idxbu = np.arange(nu, dtype=int)
    ocp.parameter_values = np.array([mpc_cfg.get("mu_nom", 1.0)], dtype=float)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver_iter_max = 400
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    return ocp
