#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np
from acados_template import AcadosOcpSolver
from ..mpc_core.models.bicycle_model import build_bicycle_model
from ..mpc_core.ocp.ocp_nominal import build_nominal_ocp
from ..config.loader import load_yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True); ap.add_argument("--tires", required=True); ap.add_argument("--mpc", required=True)
    args = ap.parse_args()
    car = load_yaml(args.car); tires = load_yaml(args.tires); mpc = load_yaml(args.mpc)
    model = build_bicycle_model(car, tires); ocp = build_nominal_ocp(model, mpc)
    _ = AcadosOcpSolver(ocp, json_file=f"{model['name']}_acados.json")
    print("Nominal solver built."); return 0
if __name__ == "__main__": sys.exit(main())
