# 20-lap benchmark

THETRACK (26.06 m), NUC5, floor `dubi`, `iqp` raceline. Control period 0.05 s
(`MPC_freq: 20`, from each algorithm's own config). Plant and MPC are the same model,
with friction as the only deliberate mismatch. 20 laps each, from `s = 20 m`, `v_x = 3 m/s`.

Measured after the duplicate-key fix, so **STMPC and LBMPC finally share the same
`a_lat`** in their combined tire constraint. Before it, NUC5's `correct_v_y_dot` was
declared twice and STMPC silently ran the kinematic approximation while LBMPC ran the
true `v̇_y` — the two "identical" models were solving different problems and the
comparison below was not valid. Solve times are wall-clock under whatever else was
running; treat them as indicative and compare only within a table row group.

```bash
python -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
```

## Matched friction (plant μ = MPC μ = 1.0)

| | lap best | lap mean | lap σ | mean \|n\| | max \|n\| | mean v_x | solve ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| **KMPC** | 7.10 | 7.100 | 0.000 | 0.020 | 0.051 | 3.70 | 1.9 | 2.5 |
| **STMPC** | 7.40 | 7.448 | 0.025 | 0.078 | 0.294 | 3.64 | 5.3 | 7.1 |
| **LBMPC** | 7.45 | 7.553 | 0.033 | 0.071 | 0.253 | 3.58 | 7.0 | 9.7 |

With the constraint mismatch removed, **STMPC is the faster of the two single-track
controllers when the model is right** — 7.448 s against LBMPC's 7.553 s. That is the
expected result: with no plant/model error there is no residual worth learning, so the
GP only costs horizon time. LBMPC still tracks tighter (mean \|n\| 0.071 vs 0.078).
Its value shows up in the next table.

**KMPC's win here is an artifact, not a result.** Its plant is its own kinematic
discrete model, which has no tire model and therefore cannot lose grip — the car is
free to corner at any speed the box constraints allow. STMPC and LBMPC are held to a
Pacejka plant. Do not read the first row as "KMPC is the better controller".

## Friction mismatch (plant μ = 0.8, MPC still assumes 1.0)

| | lap best | lap mean | mean \|n\| | max \|n\| | mean v_x | solve ms | p99 ms | GP pts |
|---|---|---|---|---|---|---|---|---|
| **STMPC** | 8.10 | 8.160 | 0.185 | 0.489 | 3.53 | 5.8 | 8.0 | — |
| **LBMPC**, no learning | 8.20 | 8.210 | 0.168 | 0.426 | 3.48 | 3.9 | 5.2 | 0 |
| **LBMPC**, learning | **7.95** | **7.963** | **0.101** | **0.331** | 3.45 | 8.5 | 11.8 | 137 |

Online GP learning buys **−3.0 % lap time and −40 % mean lateral deviation** against the
same controller with learning off, and −2.4 % / −45 % against nominal STMPC. Solver cost
roughly doubles (3.9 → 8.5 ms mean, p99 11.8) — still well inside a 50 ms budget.

So the ordering flips with the model error: STMPC wins when its model is right, LBMPC wins
when it is not. That is the whole point of the residual, and it is only visible now that
both controllers constrain the same `a_lat`.

Those learning numbers need the dictionary-gate fix in
`external/l4acados` (`fix/balanced-dictionary-gate`). With the previous admission rule the
dictionary froze at 8 points and the same run gave 8.05 s / 0.140 m — most of the available
benefit was being left on the table. See "Dictionary gate" below.

KMPC is absent from this table: its model has no friction parameter, so there is nothing
to mismatch.

## Solver status

All 20-lap runs: 100 % of solves returned 0 or 2. `status=2` is normal, not a failure —
`nlp_solver_max_iter = 3` with `SQP_WITH_FEASIBLE_QP` means the SQP is truncated by
design. STMPC returned 2 on every step, LBMPC 0 on every step, KMPC a mix.

Costs are **not** comparable across algorithms (KMPC's mean 34.3 vs STMPC's 6.3): different
cost expressions over different state vectors.

## LBMPC's own research experiments

`research/residual_validation/validation/experiments/sim_residual_cl_gpytorch.py`, under a
harsher mismatch (plant μ 0.8 vs model 1.0 **plus** a +1.0 m/s² v_x bias, 1e-5 state noise),
50-point dictionary:

```
GP training MLL loss      0.727 -> -2.086 over 200 iterations
dictionary                50 kept / 600 dropped (m_max=50, min_dist=0.15)
nominal STMPC, 3 laps     8.465 / 8.410 / 8.432 s | lateral dev 0.279 ± 0.290 m
LB-MPC, 3 laps            8.778 / 8.633 / 8.624 s | lateral dev 0.176 ± 0.199 m  (-37 %)
solve time                nominal 4.8 ms mean | LB-MPC 41.6 mean, 25.4 median, 185.5 p95
```

Held-out one-step residual RMSE, 600 pairs never in the dictionary (nominal model predicts
residual ≡ 0, so its RMSE is RMS of the target). Measured once with a throwaway harness
built on the script's own `dyn_from_ctrl` / `integrate_one_step` / `update_dictionary`;
unlike the numbers above it has not been re-run, so treat the magnitudes as indicative:

| channel | nominal | GP | |
|---|---|---|---|
| v_x | 0.058883 | 0.012886 | **−78 %** |
| v_y | 0.073895 | 0.047387 | **−36 %** |
| yaw rate | 0.052255 | 0.068322 | **+31 % worse** |
| norm | 0.107973 | 0.084139 | −22 % |

It generalises rather than memorising. The v_x win is the injected bias being recovered.
**The yaw-rate residual gets worse** — w_z carries little structured signal under a
friction-only mismatch and the shared 5-D feature GP overfits it. Real finding, not an
extraction artifact. Closed loop it still nets out positive.

The other two experiment scripts (`validate_residual_mpc.py`,
`simulate_residual_closed_loop.py`) do not run: they call `build_multitask_gp`,
`gp_predictor_from_model` and `save_gp_state`, deleted from `gp_interface.py` by race_stack
`e8e132245` (2025-11-24) without updating the callers. Dead seven months before this
extraction.

## Dictionary gate (fixed)

`OnlineLearningStrategy`'s `"balanced"` rule admitted a candidate only if its
nearest-neighbour distance exceeded *the dictionary's own smallest pairwise distance*. That
quantity never decreases when a point is admitted and is only recomputed after an eviction —
and evictions happen only once the dictionary is full. Below capacity the bar was therefore
frozen at whatever gap the first two admitted samples happened to have.

Replaying the rule over one fixed 20-lap trajectory, subsampled so only the spacing changes:

| sample rate | 1st-pair gap | final dictionary |
|---|---|---|
| 50 Hz | 0.263 | 137 |
| 25 Hz | 0.563 | 50 |
| 16.7 Hz | 0.832 | 30 |
| 10 Hz | 1.192 | 17 |

And at one fixed rate, 30 vs 94 points for identical data depending only on the starting
offset. At 20 Hz the real controller settled at **8** points against a `max_num_data_points`
of 200, because its first two residual pairs land in the startup transient. Distances were
also taken over raw features, so admission was dominated by whichever channel had the widest
numeric range.

Fixed upstream in the fork: admit iff the candidate is at least `min_dist` from every
dictionary member, as a per-feature RMS distance in units of each feature's standard
deviation (running Welford estimate, or `feature_scales` if supplied). Eviction from the
closest pair when full is unchanged. Default `min_dist=0.15` gives 127–137 points on this
track at 20 Hz. `test/gpytorch_model/test_online_learning_gate.py` in the fork covers
rate/offset independence, scale invariance, duplicate rejection and the per-admission bound.

## Reference values

Single STMPC solve, NUC5/dubi, THETRACK `iqp`, x0 = `(s=20, n=0, ψ=0, v_x=3)`, one fixed
22-entry p on every stage — the parity oracle used throughout the refactor:

```
cost 32.86543961175853    status 2    n_x 8    N 40
```

It was `34.69739908591064` before NUC5's `correct_v_y_dot` was fixed. race_stack's
pre-extraction STMPC reproduces it bit-for-bit (`x_traj` max|diff| = 0.0), which is what
keeps the extraction honest.

## ZoRO, exercised for the first time

Every shipped fixture sets `stochastic_mode: disabled`, so until now no configuration in the
repo had ever run `add_zoro_description`. Driving all three modes for 200 closed-loop steps
under a friction mismatch (plant μ 0.8, MPC 1.0), on THETRACK/NUC5:

| mode | `fdbk_K_mat` | 200 steps | verdict |
|---|---|---|---|
| `open_loop` | n/a | leaves the track at step 162, n = 1.66 m | no ancillary feedback, as designed |
| `lqr` | \|K\| max 13.06 | all 200 solves status 0, n = −0.004 m | works |
| `riccati` | **\|K\| max 0** | leaves the track at step 162, n = 1.66 m | **broken** |

`riccati` fails *identically* to `open_loop`, down to the same step and the same lateral
deviation, and its feedback gain comes out all zeros. `add_zoro_description` sets
`fdbk_K_mat = np.zeros((nu, nx))` and `feedback_optimization_mode =
"RICCATI_CONSTANT_COST"`, expecting acados to derive the gain from `riccati_Q_const` /
`riccati_R_const`. It does not, so riccati silently degrades to open loop.

`lbmpc/README.md` calls riccati "usually the most principled option and the default
stochastic choice". On this acados/l4acados pin it is the worst of the three. **`lqr` is the
only stochastic mode that currently works.**

Not caused by the state permutation: the whole block is enum-driven, and none of these modes
had ever been executed. One genuine ordering bug did live there and is fixed — the R-matrix
regularisation wrote `R[InputIndex.X, InputIndex.X]` but read `R[0, 0]`/`R[1, 1]`, so under
the old input order it put `qjerk` into the ddelta slot.

## Open questions

- **acados' `code_export_directory` defaults to a cwd-relative `c_generated_code`**, so two
  runs started from different working directories quietly inherit each other's compiled
  solvers. A stale `c_generated_code/` left at the repo root was enough to make a freshly
  built LBMPC load a solver from an older model and die with
  `Spatialbicycle_model_acados_sim_update_params: trying to set 22 parameters for external
  functions. External function has 14 parameters.` The directory is gitignored but not
  cleaned, and both the OCP and the sim model are named `Spatialbicycle_model`, so nothing
  in the filename says which build it came from. Delete `c_generated_code/` before trusting
  any parity result, and consider pinning `code_export_directory` per controller.

- **LBMPC computes its residuals against a zero input on the car.**
  `lb_mpc.py` sets `_prev_input_for_residual = np.zeros(2)` with `self.u0.copy()` commented
  out beside it, so `r = x_k − f(x_{k−1}, 0)` attributes the whole effect of the applied
  input to model error. race_stack `aeb5101d9` (2026-02-22, "Improvement in residual
  calculation validated") — pre-existing, and the sim adapter passes the real `u0` instead.
  Worth a decision before the next car run.
- `StateIndex.INPUT_JERK` / `INPUT_DSTEERING` should be restored in
  `forzaeth_mpc/utils/indices.py` (and race_stack's copy); they were dropped by a merge
  revert, `94ea48a79`.
