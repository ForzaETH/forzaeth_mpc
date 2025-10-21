#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np, casadi as ca
from acados_template import AcadosOcpSolver
from ..mpc_core.models.bicycle_model import build_bicycle_model
from ..mpc_core.ocp.ocp_nominal import build_nominal_ocp
from ..mpc_core.utils.integrators import integrate_open_loop
from ..validation.data.dataset_one_step import build_dataset_one_step
from ..config.loader import load_yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True); ap.add_argument("--tires", required=True); ap.add_argument("--mpc", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()
    car = load_yaml(args.car); tires = load_yaml(args.tires); mpc = load_yaml(args.mpc)
    model = build_bicycle_model(car, tires); ocp = build_nominal_ocp(model, mpc)
    solver = AcadosOcpSolver(ocp, json_file=f"{model['name']}_acados.json")
    x0 = np.zeros((model["nx"],), dtype=float); solver.set(0, "x", x0); status = solver.solve()
    if status != 0: raise RuntimeError(f"acados solve failed: {status}")
    H = ocp.dims.N; Xp = np.stack([solver.get(j,"x") for j in range(H+1)], axis=0); Up = np.stack([solver.get(j,"u") for j in range(H)], axis=0)
    x = model["x"]; u = model["u"]; p = model["p"]; f = model["f_expl"]; f_fun = ca.Function("f", [x,u,p], [f])
    dt = float(mpc.get("dt", 0.04)); mu_nom = float(mpc.get("mu_nom", 1.0)); bias = np.array([0.0,0.0,0.05])
    def write_u(xv, uv): return xv.copy()
    def dyn_nominal(t, xv): return np.array(f_fun(xv, np.zeros_like(Up[0]), np.array([mu_nom]))).reshape(-1)
    def dyn_actual(t, xv): 
        dx = dyn_nominal(t, xv).copy(); dx[3]+=bias[0]; dx[4]+=bias[1]; dx[5]+=bias[2]; return dx
    nominal_step = lambda x,u,DT: integrate_open_loop(dyn_nominal, x, u[None,:], DT, write_u)[-1]
    actual_step  = lambda x,u,DT: integrate_open_loop(dyn_actual,  x, u[None,:], DT, write_u)[-1]
    Z,Y = build_dataset_one_step(nominal_step, actual_step, Xp, Up, dt); np.savez(args.out_npz, Z=Z, Y=Y); print(f"Saved {args.out_npz}"); return 0
if __name__ == "__main__": sys.exit(main())
