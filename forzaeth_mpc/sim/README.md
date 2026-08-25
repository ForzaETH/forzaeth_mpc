# forzaeth_mpc.sim

This simulator drives the same controller objects that the car drives. It builds them with
their own constructors, through three adapters: `ControllerMPC` for STMPC, `ControllerKMPC`
for KMPC and `ControllerLBMPC` for LB-MPC. It reads the same configuration schemas and the
same exported tracks.

The simulator does not use ROS. It imports no `rospy`, it needs no `roscore`, and it uses no
message types.

The plant integrates the dynamics of the controller under test. Two kinds of model error are
available:

- Friction. `--mu_track` sets the friction of the plant, and `--mu_mpc` sets the friction that
  the controller assumes. Give them different values. `run_laps`, `run_single_rollout` and
  `analyze_cost_vs_mu` accept both. KMPC ignores them, because a kinematic model has no tires.
- Noise. `run_single_rollout` accepts `--noise_std`, and
  `noise_robustness_analysis` accepts `--noise_std_list`. Both add noise to the forward speed,
  the lateral speed and the yaw rate.

## Directory

```
forzaeth_mpc/sim/
├─ cli/             # runnable entry points
│  ├─ run_laps.py                # N laps, fixed metrics, all three controllers
│  ├─ run_single_rollout.py      # one rollout, with plots
│  ├─ analyze_cost_vs_mu.py      # results against the assumed friction
│  ├─ tune_mpc_weights.py        # weight search with Optuna
│  └─ noise_robustness_analysis.py
│
├─ core/            # the simulation engine
│  ├─ scenario.py   # load the configs and the track, build a controller
│  ├─ controllers.py# the three adapters
│  ├─ plant.py      # integrate the dynamics of the controller
│  └─ rollout.py    # close the loop
│
├─ tracks.py        # read a global_waypoints.json into an (N,7) array
├─ waypoints.py     # the column layout of that array
├─ metrics/         # lap times, tracking error, solver results
├─ plotting/        # figures
└─ utils/           # integrators
```

## How to run the commands

Every argument has a default, so each command runs without flags. Run the commands from the
root of the repository, or from any directory after you install the package.

```bash
python3 -m forzaeth_mpc.sim.cli.run_single_rollout
```

The plots need a display. Without one, set `MPLBACKEND=Agg`. The commands then write no
figures, but they still print their results.

A lap of THETRACK takes approximately 377 steps at `--dt 0.02`. Keep `--steps` well above
that number. A run that completes no lap reports no lap time, and the metrics that need one
become useless.

### run_laps.py

This command drives one or more controllers for a given number of laps. It writes a fixed set
of metrics to a json file. `BENCHMARK.md` reports these numbers.

```bash
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
```

The control period comes from `MPC_freq` in the configuration of each controller, not from
the command. The plant of KMPC is a discrete model, and it advances exactly one control
period for each call. Use `--dt` to override the period.

`--dump_states` writes the state and the input of every step to a `.npy` file. Use it to
compare two versions of the code.

`--video` records the run to `<out>/<algo>_run.mp4`. Each frame draws the track boundaries,
the last few seconds of the driven path, the horizon that the controller predicts at that
step, and the speed. The predicted horizon is the part that a saved figure cannot show.

```bash
python3 -m forzaeth_mpc.sim.cli.run_laps --algo stmpc --laps 2 --video --out /tmp/mpc_laps
```

The file needs no display, because the figure is never shown. The recorder uses ffmpeg if it
finds it, and the binary of `imageio-ffmpeg` if ffmpeg is not on the path. Without either it
writes an animated gif, which is about 20 times larger.

`--video_every N` records one frame every N control steps. `--video_fps` sets the rate of the
file. The defaults, 1 and 20, give real time at 20 Hz control. Two laps make a 15 s file of
about 3 MB, so record two laps and not twenty.

### run_single_rollout.py

This command drives STMPC once, and it plots the result.

```bash
python3 -m forzaeth_mpc.sim.cli.run_single_rollout \
  --track THETRACK --racecar_version NUC5 --floor dubi --refline iqp \
  --dt 0.02 --steps 600 --integrator rk4 \
  --mu_track 1.00 --mu_mpc 1.00 \
  --s0 20 --n0 0.0 --psi0 0.0 --vx0 3.0 \
  --vis_spaghetti_predictions_vs_actual \
  --vis_state_prediction_vs_actual \
  --vis_target_vel_actual
```

The figures show the predicted path against the driven path, the target speed against the
measured speed, and a summary of the run.

![One rollout of STMPC](doc/single_rollout.png)

### analyze_cost_vs_mu.py

This command shows what happens when the friction of the track differs from the friction that
the controller assumes.

```bash
python3 -m forzaeth_mpc.sim.cli.analyze_cost_vs_mu \
  --track THETRACK --racecar_version NUC5 --floor dubi \
  --dt 0.02 --steps 700 --mu_track 1.00 \
  --mu_mpc_list 0.85 0.90 0.95 1.00 1.05 1.10 1.15
```

It drives one run for each value in `--mu_mpc_list`. It then compares the lap time, the cost
and the tracking error of those runs.

<p align="center">
  <img src="doc/mu_mismatch_analysis.png" alt="Results against the assumed friction" width="500">
</p>

### tune_mpc_weights.py

This command searches for cost weights with Optuna. The objective adds four terms: the lap
time, the mean solver cost, the tracking error and the speed error. `--w_lt`, `--w_cost`,
`--w_abs_n` and `--w_abs_v` set the weight of each term.

```bash
python3 -m forzaeth_mpc.sim.cli.tune_mpc_weights \
  --track THETRACK --racecar_version NUC5 --floor dubi \
  --dt 0.02 --steps 900 --trials 40 \
  --mu_track 1.00 --mu_mpc 1.00 \
  --w_lt 1.0 --w_cost 0.0 --w_abs_n 0.1 --w_abs_v 0.0 \
  --vis_parallel
```

A trial that completes no lap gets a large penalty, and it prints a message. If most trials
print that message, increase `--steps`. All the trials then get the same penalty, and the
search learns nothing.

![The sampled weights against the metrics](doc/parallel_cost_plot.png)

### noise_robustness_analysis.py

This command adds noise to the state of the plant, and it repeats the run for each level in
`--noise_std_list`. `--seed` fixes the random numbers, so a run repeats exactly.

```bash
python3 -m forzaeth_mpc.sim.cli.noise_robustness_analysis
```

## How to add a command

1. Write a new file in `cli/`.
2. Import the engine:

   ```python
   from forzaeth_mpc.sim.core.controllers import ControllerMPC
   from forzaeth_mpc.sim.core.plant import Plant
   from forzaeth_mpc.sim.core.rollout import RolloutRunner
   from forzaeth_mpc.sim.core.scenario import Scenario
   ```
3. Build the run, then use the modules in `metrics/` and `plotting/` for the results.

Do not change the controllers to add a test. `Scenario` builds each one through its real
constructor, and the adapters in `controllers.py` hold everything that the simulator needs.

## Suggested order of work

1. Drive one rollout, and confirm that the controller is stable.
2. Sweep the friction, and see how much error the controller accepts.
3. Search for better weights.
4. Drive 20 laps, and compare the metrics against `BENCHMARK.md`.
