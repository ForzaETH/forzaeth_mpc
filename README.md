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

STMPC and LB-MPC share one model, `forzaeth_mpc/stmpc/bicycle_model.py`. The state order is
`[s, n, theta, v_x, v_y, yaw_rate, delta, accel]`. The input order is `[ddelta, jerk]`. Use
`forzaeth_mpc.utils.indices` to read either one. Do not use integer literals.

LB-MPC learns `x_{k+1} = f_nominal(x_k, u_k) + B*mu_GP(z_k)` while it drives. The residual is
not part of the CasADi model. l4acados applies it outside the model. Read
[`forzaeth_mpc/lbmpc/README.md`](forzaeth_mpc/lbmpc/README.md) for the parameters and the
workflow.

## Simulator

`forzaeth_mpc/sim/` drives the same controller objects that the car drives. It closes the
loop against the controller's own dynamics. It does not use ROS.

The plant accepts a different friction value than the MPC assumes. It can also add noise to
the state. Use both to introduce model error on purpose.

```bash
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
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

python3 -m pip install -e ".[sim]"                     # controllers + simulator
./install/install_l4acados.sh               # only for LB-MPC
python3 -m pip install -e ".[sim,lbmpc]"
```

`install_l4acados.sh` installs two pinned forks, `naefjo/gpytorch` and
`naefjo/linear_operator`. The exo backend of l4acados needs them. The script replaces any
copy from PyPI. LB-MPC does not work with the released `gpytorch`.

## Configuration

Pydantic validates every configuration file. A missing key or an extra key stops the load.
No value falls back to a default without a message.

The schemas are in `forzaeth_mpc/config/`. The values are in `config/<CAR>/*.yaml`.

```python
from forzaeth_mpc.config import load_STMPC_config_ros, load_car_config_ros
cfg = load_STMPC_config_ros("NUC5")
```

The loaders reject a file that declares the same key twice. Plain `yaml.safe_load` keeps the
last value and reports nothing. That behaviour hid a wrong tire constraint on two cars.

The `config/` directory holds test fixtures. It is large enough to run the simulator. It is
not the source of truth for a car. On a car, point the loaders at race_stack:

```bash
export FORZAETH_MPC_CONFIG=$(rospack find stack_master)/config
```

The fixtures do not repeat the tuning values of the cars. Car `NUC5` with floor `dubi` is
the combination that the simulator uses by default, and the only complete one.

`tracks/*.json` are `global_waypoints.json` files from the global planner of race_stack. Set
`FORZAETH_MPC_TRACKS` to read them from a different directory.

## Use from race_stack

Clone the package beside the other external repositories, then install it. The ROS nodes of
race_stack import from it, and `pbl_config` re-exports the configuration schemas.

```bash
cd ~/catkin_ws/src/external && git clone --recurse-submodules <this repo> forzaeth_mpc
python3 -m pip install -e "forzaeth_mpc[sim,lbmpc]"
```

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
- **`install/install_l4acados.sh` writes a `.pth` file that points to
  `external/l4acados/external`.** The current l4acados commit has no such directory. Python
  ignores the path, so the file has no effect.
- **A built wheel does not contain `config/` or `tracks/`.** Both directories are outside the
  importable package. Install the package from a clone.
