#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, importlib.util
from ..mpc_core.models.bicycle_model import build_bicycle_model
from ..mpc_core.ocp.ocp_nominal import build_nominal_ocp
from ..mpc_core.residual.residual_models import make_residual_model_from_gp
from ..mpc_core.ocp.ocp_residual import build_residual_mpc
from ..mpc_core.utils.indices import IDX
from ..config.loader import load_yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True); ap.add_argument("--tires", required=True); ap.add_argument("--mpc", required=True)
    ap.add_argument("--gp_module", required=True)  # module file that defines predict_mean(Z)->(N,3)
    args = ap.parse_args()
    car = load_yaml(args.car); tires = load_yaml(args.tires); mpc = load_yaml(args.mpc)
    model = build_bicycle_model(car, tires); ocp = build_nominal_ocp(model, mpc)
    spec = importlib.util.spec_from_file_location("gp_mod", args.gp_module); gp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp_mod); predict_mean = gp_mod.predict_mean
    residual_model = make_residual_model_from_gp(predict_mean, {"vx":IDX["vx"],"vy":IDX["vy"],"wz":IDX["wz"]}, model["nx"])
    l4a_solver = build_residual_mpc(ocp, residual_model); print("Residual-aware solver built."); return 0
if __name__ == "__main__": sys.exit(main())
