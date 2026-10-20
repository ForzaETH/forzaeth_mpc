# forzaeth_mpc

Model predictive controllers for the ForzaETH F1TENTH car. The package also contains a
simulator that runs the controllers without ROS.

The code comes from the
[race_stack](https://git.ee.ethz.ch/pbl/research/f1tenth/race_stack), with its git history.
The ROS nodes stay in race_stack. This package holds the control algorithms, the
configuration schemas and the simulator.

## Controllers

| Module | Description |
|---|---|
| `forzaeth_mpc/kmpc/` | Kinematic bicycle MPC. Five states. No tire model. |
| `forzaeth_mpc/stmpc/` | Single-track MPC with a Pacejka tire model. Eight states. |
| `forzaeth_mpc/lbmpc/` | The single-track MPC plus a learnt residual, and optional ZoRO constraint tightening. |
| `forzaeth_mpc/tompc/` | Time-optimal MPC. The single-track model reformulated over distance. Nine states. |

STMPC, LB-MPC and TOMPC share one model, `forzaeth_mpc/stmpc/bicycle_model.py`. The state
order is `[s, n, theta, v_x, v_y, yaw_rate, delta, accel]`. The input order is
`[ddelta, jerk]`. Use `forzaeth_mpc.utils.indices` to read either one. Do not use integer
literals.

TOMPC integrates over the path coordinate instead of over time, and minimises the time the
horizon takes rather than the error against a speed profile. It appends elapsed time to the
state, so its `x[:8]` is still a single-track state. Read
[`forzaeth_mpc/tompc/README.md`](forzaeth_mpc/tompc/README.md) before tuning it: its
acceleration limits are load-bearing where the other controllers' are not.

LB-MPC learns `x_{k+1} = f_nominal(x_k, u_k) + B*mu_GP(z_k)` while it drives. The residual is
not part of the CasADi model. l4acados applies it outside the model. Read
[`forzaeth_mpc/lbmpc/README.md`](forzaeth_mpc/lbmpc/README.md) for the parameters and the
workflow.

Each controller package has the same shape:

```
controller.py       the controller class
acados_settings.py  the OCP
online_params.py    the online parameter vector
bicycle_model.py    the model  (lbmpc imports stmpc's)
```

LB-MPC carries three more modules, because it owns things the other two do not:
`solver_artifacts.py` for the generated solver on disk and the format of a saved model,
`residual_model.py` for the Gaussian process, and `offline_viz_logger.py` for the CSV log.

## Simulator

`forzaeth_mpc/sim/` drives the same controller objects that the car drives. It closes the
loop against the controller's own dynamics. It does not use ROS.

The plant accepts a different friction value than the MPC assumes. It can also add noise to
the state. Use both to introduce model error on purpose.

```bash
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc tompc --laps 20 --plot
python3 -m forzaeth_mpc.sim.cli.run_single_rollout      # one rollout, with plots
python3 -m forzaeth_mpc.sim.cli.analyze_cost_vs_mu      # friction sweep
python3 -m forzaeth_mpc.sim.cli.tune_mpc_weights        # weight search with Optuna
python3 -m forzaeth_mpc.sim.cli.noise_robustness_analysis
```

Each argument has a default. All five commands run without flags.
[`forzaeth_mpc/sim/README.md`](forzaeth_mpc/sim/README.md) describes them.
[`BENCHMARK.md`](BENCHMARK.md) gives the measured results.

## Installation

Pip cannot resolve acados or the GP libraries. Use the two shell scripts.

```bash
git clone --recurse-submodules <this repo>
cd forzaeth_mpc

./install/install_acados.sh ~/acados        # acados v0.4.4 + acados_template
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH

python3 -m pip install git+https://github.com/ForzaETH/forza_config.git   # config schemas
python3 -m pip install -e ".[sim]"                     # controllers + simulator
./install/install_l4acados.sh               # only for LB-MPC
python3 -m pip install -e ".[sim,lbmpc]"
```

`install_l4acados.sh` installs two pinned forks, `naefjo/gpytorch` and
`naefjo/linear_operator`. The exo backend of l4acados needs them. The script replaces any
copy from PyPI. LB-MPC does not work with the released `gpytorch`.

It installs the CPU build of `torch` by default, so pip does not pull ~2.5 GB of CUDA
wheels on a machine without a GPU. On a GPU machine, set the index first:
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./install/install_l4acados.sh`.

## Configuration

Pydantic validates every configuration file. A missing key or an extra key stops the load,
and so does a key declared twice in the same file. No value falls back to a default.

The MPC schemas are in `forzaeth_mpc/config/`. The car, tire and trailing schemas come from
[`forza_config`](https://github.com/ForzaETH/forza_config), so both packages read the same
files the same way. This package carries no yaml: the values live in race_stack, in
`stack_master/config/<CAR>/`. Its ROS nodes read them with `forza_config_ros`.

The simulator and the module self-tests read a car by name from `FORZAETH_MPC_CONFIG`:

```bash
export FORZAETH_MPC_CONFIG=<race_stack>/stack_master/config
```

```python
from forzaeth_mpc.config import load_STMPC_config_ros, load_car_config_ros
cfg = load_STMPC_config_ros("NUC5")
```

Car `NUC5` with floor `dubi` is the combination that the simulator uses by default.

`--track` takes the name of a `global_waypoints.json` from the global planner of race_stack, or
a path to any such file. Point it at the maps of race_stack:

```bash
export FORZAETH_MPC_TRACKS=<race_stack>/stack_master/maps
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc --track the_grand_tour --laps 2
```

`FORZAETH_MPC_TRACKS` accepts both layouts: `<name>.json` and `<name>/global_waypoints.json`.
Use `--refline` to pick the raceline in the file. Every map of race_stack holds `iqp` and
`sp`.

## Use from race_stack

race_stack has this package as a submodule at `controller/forzaeth_mpc` and builds it with
colcon (`package.xml` only lets colcon and rosdep find it; the metadata is in `setup.py`
because the ROS humble images pin setuptools 58, which ignores pyproject's `[project]`).
Its ROS nodes import the controllers from here.

## Limitations

- **ZoRO `riccati` does not work.** Its feedback gain is zero, and the controller behaves
  like `open_loop`. Use `lqr`. `BENCHMARK.md` gives the measurements.
- **LB-MPC computes each residual against a zero input.** `lb_mpc.py` sets
  `_prev_input_for_residual` to `np.zeros(2)`. Every residual therefore charges the whole
  effect of the applied input to model error. Decide this before the next run on a car.
- **acados writes generated code to a path relative to the working directory.** Two runs
  from different directories can load each other's compiled solver. The three controllers
  now use different model names, which removes the collision between them. Delete
  `c_generated_code/` if a result looks wrong.
- **`research/residual_validation/` is frozen.** It keeps its own copy of the single-track
  model, at the earlier state order and the 14-entry parameter vector. The published GP
  numbers come from that copy. Do not change it to use the shared model.
