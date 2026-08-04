# 20-lap benchmark

THETRACK (26.06 m), NUC5, floor `dubi`, `iqp` raceline. Control period 0.05 s
(`MPC_freq: 20`, from each algorithm's own config). Plant and MPC are the same model,
with friction as the only deliberate mismatch. 20 laps each, from `s = 20 m`, `v_x = 3 m/s`.

```bash
python -m forzaeth_mpc.sim.cli.run_laps --algo stmpc kmpc lbmpc --laps 20 --plot
```

## Matched friction (plant μ = MPC μ = 1.0)

| | lap best | lap mean | lap σ | mean \|n\| | max \|n\| | mean v_x | solve ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| **KMPC** | 7.10 | 7.10 | 0.000 | 0.020 | 0.051 | 3.70 | 2.6 | 5.2 |
| **STMPC** | 7.45 | 7.47 | 0.029 | 0.082 | 0.286 | 3.64 | 6.9 | 8.5 |
| **LBMPC** | 7.45 | 7.55 | 0.033 | 0.071 | 0.253 | 3.58 | 9.2 | 20.9 |

**KMPC's win here is an artifact, not a result.** Its plant is its own kinematic
discrete model, which has no tire model and therefore cannot lose grip — the car is
free to corner at any speed the box constraints allow. STMPC and LBMPC are held to a
Pacejka plant. Do not read the first row as "KMPC is the better controller".

## Friction mismatch (plant μ = 0.8, MPC still assumes 1.0)

| | lap best | lap mean | mean \|n\| | max \|n\| | mean v_x | solve ms | GP pts |
|---|---|---|---|---|---|---|---|
| **STMPC** | 8.15 | 8.19 | 0.189 | 0.469 | 3.52 | 6.9 | — |
| **LBMPC**, no learning | 8.20 | 8.21 | 0.168 | 0.426 | 3.48 | 4.5 | 0 |
| **LBMPC**, learning | **7.95** | **7.96** | **0.101** | **0.331** | 3.51 | 9.9 | 137 |

Online GP learning buys **−3.0 % lap time and −40 % mean lateral deviation** against the
same controller with learning off. Solver cost roughly doubles (4.5 → 9.9 ms mean,
p99 13.1 ms) — still well inside a 50 ms budget.

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

## Open questions

- **LBMPC computes its residuals against a zero input on the car.**
  `lb_mpc.py` sets `_prev_input_for_residual = np.zeros(2)` with `self.u0.copy()` commented
  out beside it, so `r = x_k − f(x_{k−1}, 0)` attributes the whole effect of the applied
  input to model error. race_stack `aeb5101d9` (2026-02-22, "Improvement in residual
  calculation validated") — pre-existing, and the sim adapter passes the real `u0` instead.
  Worth a decision before the next car run.
- `StateIndex.INPUT_JERK` / `INPUT_DSTEERING` should be restored in
  `forzaeth_mpc/utils/indices.py` (and race_stack's copy); they were dropped by a merge
  revert, `94ea48a79`.
