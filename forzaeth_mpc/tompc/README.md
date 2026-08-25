# Time-optimal MPC (TOMPC)

A minimum-time controller. It does not track a speed profile. It asks for the fastest
way through the next `s_horizon` metres and drives that.

## The formulation

The controller follows

- R. Verschueren, M. Zanon, R. Quirynen, M. Diehl, *Time-optimal race car driving using
  an online exact Hessian based nonlinear MPC algorithm*, ECC 2016 — the spatial
  reformulation and the minimum-time objective,
- D. Kloeser et al., *NMPC for Racing Using a Singularity-Free Path-Parametric Model
  with Obstacle Avoidance*, IFAC 2020 — the path-parametric single-track model, which
  is the model `forzaeth_mpc/stmpc/` already runs.

The vehicle model is not a new one. `spatial_model.py` calls `single_track_dynamics()`
out of `stmpc/bicycle_model.py`, the same function STMPC and LB-MPC use.

Change the independent variable from time to the path coordinate `s`. For a state `x`
with time derivative `xdot`,

    dx/ds = xdot / sdot,   sdot = (v_x cos(theta) - v_y sin(theta)) / (1 - kappa * n)

and elapsed time joins the state with `dt/ds = 1/sdot`. The horizon now covers a fixed
distance, and the time that distance takes is `sum_i ds / sdot_i`. That sum is the stage
cost. Minimising it minimises lap time directly, instead of minimising the tracking
error against a speed profile that was a precomputed guess at it.

The state is `[s, n, theta, v_x, v_y, yaw_rate, delta, accel, t]`, the input is
`[ddelta, jerk]`. The first eight entries are the single-track layout unchanged, so
`x[:8]` of a spatial state is a single-track state — read them through
`SpatialStateIndex`, never as integers.

## What it costs you

**Two singularities, and they fail differently.** `1/sdot` blows up as `sdot -> 0` and
goes to *zero* as `sdot -> infinity`. The second is the dangerous one: `sdot` has
`1 - kappa * n` in its denominator, so a large enough lateral deviation towards the
centre of curvature makes the predicted traversal time vanish. The track boundary rules
that state out, but it is a *soft* constraint, so left alone the solver buys the
singularity and pays the slack. The hard bound `sdot <= s_dot_max` is what closes it.
That row is deliberately not in `idxsh`. `v_min > 0` and `sdot >= s_dot_min` handle the
other end.

**The acceleration limits have to be real.** STMPC ships `a_min`/`a_max` at ∓15 and
`alat_max` at 15, with a note in its own yaml that the numbers are deliberately too
high. A controller that tracks a speed profile never asks for its limit, so a loose one
is harmless. This one sits on the limit every cycle. With STMPC's numbers it asks for
15 m/s² of lateral from a tire that delivers 8, and the car spins on the first tight
corner. `alat_max`, `a_min`, `a_max` and `v_max` are now load-bearing tuning.

**The integrator needs more substeps.** A distance grid means the *time* a node spans
depends on speed: 0.3 m is 37 ms at 8 m/s but 600 ms at 0.5 m/s. The Pacejka model is
stiff enough that RK4 diverges over the latter, and it does not fail quietly — it puts a
NaN in the QP and HPIPM returns status 3 on the first QP of every call.
`sim_method_num_steps` is 10 where the time-domain controllers use 3, and `v_min` bounds
the worst case.

**It cannot launch from standstill.** The model is singular at `v_x = 0`. Below `v_min`
there is no OCP to solve, and `controller_manager` falls back to MAP, as it does for a
failed solve.

## What you get back

Delay compensation is exact rather than estimated. STMPC applies the command from a
fixed `steps_delay` node, which is only correct at one speed. Here the elapsed time is a
state, so `delay_node()` returns the first node whose clock has passed `t_delay` — right
while braking and accelerating too.

`horizon_time` is the objective in seconds, published on
`/mpc_controller/horizon_time`. Unlike the raw solver cost it is comparable between
tunings, so it is the number to watch while tuning.

## Modes

Minimum time and holding a gap contradict each other, so the controller switches rather
than blending. `online_params.racing_weights()` zeroes `q_v`; `trailing_weights()`
zeroes `q_time` and hands the speed to the gap controller. Overtaking keeps the time
term and takes the lateral offset from the planner.

## Tuning

The time term of the cost is exactly `qtime * (horizon traversal time in seconds)`, so
every other weight reads as "how many qtime-seconds is this worth". Only the ratios
change the optimum; `qtime` defaults to 100 rather than 1 so the terms come out order 1,
which the QP prefers.

Keep `qn` low. The time-optimal line is not the centre line, and a high `qn` just tracks
the raceline again, which is the thing this controller exists to stop doing.

## Measured

`python3 -m forzaeth_mpc.sim.cli.run_laps --algo tompc stmpc kmpc --laps 10`, THETRACK,
NUC5/dubi, each controller on its own shipped tuning:

| algo | best lap | mean lap | max abs n | solver ok | solve ms (mean / p99) |
|---|---|---|---|---|---|
| tompc | 6.95 s | 7.01 s | 0.41 m | 100 % | 9.5 / 15.7 |
| kmpc | 7.10 s | 7.10 s | 0.05 m | 100 % | 1.5 / 2.0 |
| stmpc | 7.40 s | 7.46 s | 0.29 m | 100 % | 3.8 / 5.1 |

TOMPC is the fastest of the three and the one that wanders furthest from the centre
line, which is what a controller that is not tracking the raceline should do. Treat the
margin as indicative and not as a measurement of the algorithm: STMPC and KMPC carry
tunings the team has iterated on, TOMPC carries a first pass.
