# forzaeth_mpc

Model predictive controllers for the ForzaETH F1TENTH platform, plus a ROS-free
simulator to test them.

Extracted from the [race_stack](https://git.ee.ethz.ch/pbl/research/f1tenth/race_stack)
(`controller/mpc` + `controller/lbmpc`) with history intact. The ROS nodes that drive these
controllers stay in race_stack; this repo holds the control algorithms, their configuration
schemas, and the offline simulator.

## Controllers

| Module | What it is |
|---|---|
| `forzaeth_mpc/kmpc/` | Kinematic bicycle MPC |
| `forzaeth_mpc/stmpc/` | Single-track MPC with Pacejka tire model |
| `forzaeth_mpc/lbmpc/` | Learning-based MPC: GP residual on the single-track model, optional ZoRO stochastic constraint tightening |

LB-MPC learns `x_{k+1} = f_nominal(x_k, u_k) + B·μ_GP(z_k)` online. See
[`forzaeth_mpc/lbmpc/README.md`](forzaeth_mpc/lbmpc/README.md) for the parameters that matter
and the `learn` → `use_learnt_model` → `stochastic_mode` workflow.

## Simulator

`forzaeth_mpc/sim/` runs the **same controller objects** the car runs, closed-loop, with no ROS.
The plant uses the same dynamics as the MPC plus variable friction μ and injectable noise, so a
μ mismatch or a noisy state estimate can be studied deliberately.

```bash
python -m forzaeth_mpc.sim.cli.run_single_rollout      # closed-loop rollout + plots
python -m forzaeth_mpc.sim.cli.analyze_cost_vs_mu      # friction-mismatch sweep
python -m forzaeth_mpc.sim.cli.tune_mpc_weights        # Optuna cost-weight autotuning
python -m forzaeth_mpc.sim.cli.noise_robustness_analysis
```

Every argument has a default, so all four run with no flags. Details and plots:
[`forzaeth_mpc/sim/README.md`](forzaeth_mpc/sim/README.md).

## Install

acados and the GP stack are not pip-resolvable — they need the shell installers.

```bash
git clone --recurse-submodules <this repo>
cd forzaeth_mpc

./install/install_acados.sh ~/acados        # acados v0.4.4 + acados_template
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH

pip install -e ".[sim]"                     # controllers + simulator
./install/install_l4acados.sh               # only if you want LB-MPC
pip install -e ".[sim,lbmpc]"
```

`install_l4acados.sh` pins the `naefjo/gpytorch` and `naefjo/linear_operator` forks that
l4acados' exo backend needs, and force-reinstalls over any PyPI copies. Plain `pip install
gpytorch` will not work for LB-MPC.

## Configuration

Config is pydantic-validated, not read as loose dicts — a missing or extra key is a hard error,
not a silent default. Schemas live in `forzaeth_mpc/config/`, values in `config/<CAR>/*.yaml`.

```python
from forzaeth_mpc.config import load_STMPC_config_ros, load_car_config_ros
cfg = load_STMPC_config_ros("NUC5")
```

`config/` in this repo is a **test fixture set** — enough to run the simulator, not the source of
truth for any real car. On the car, point the loaders at race_stack instead:

```bash
export FORZAETH_MPC_CONFIG=$(rospack find stack_master)/config
```

Tuning values are deliberately not duplicated here. `NUC5` + floor `dubi` is the combination the
simulator defaults to and the one guaranteed complete.

Tracks: `tracks/*.json` are `global_waypoints.json` files exported by race_stack's global planner.
Override the directory with `FORZAETH_MPC_TRACKS`.

## Use from race_stack

Clone next to the other external repos and pip install; race_stack's ROS nodes import from here.

```bash
cd ~/catkin_ws/src/external && git clone --recurse-submodules <this repo> forzaeth_mpc
pip install -e "forzaeth_mpc[sim,lbmpc]"
```

## Known rough edges

- `research/residual_validation/` is the LB-MPC validation code lifted from
  `controller/lbmpc/test`. It carries its own copies of the single-track MPC, the bicycle model
  and an acados settings module, and its own closed-loop residual simulator — it substantially
  duplicates `forzaeth_mpc/sim/`. It is kept unwired and unmodified until an LB-MPC adapter for
  `sim/core/controllers.py` is verified against a working l4acados install; then most of it
  should be deleted.
- `bicycle_model.py` exists three times (`stmpc/`, `lbmpc/`, `research/`) with three different
  contents, and the single-track MPC twice. The `lbmpc` variant carries the residual/`B`-matrix
  hooks, so these are not blind-mergeable.
- `install/install_l4acados.sh` writes an `l4acados_external.pth` pointing at
  `external/l4acados/external`, a directory the current l4acados pin does not have. Harmless —
  Python ignores non-existent `.pth` targets — but it means the file is doing nothing.
- Packaging assumes a git clone: `config/` and `tracks/` sit outside the importable package, so a
  built wheel would not carry them. Fine for `pip install -e`, not for a registry install.
