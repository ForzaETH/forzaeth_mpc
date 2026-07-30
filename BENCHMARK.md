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
| **LBMPC**, learning | 8.05 | 8.06 | 0.140 | 0.400 | 3.49 | 9.4 | 8 |

Online GP learning buys **−1.8 % lap time and −17 % mean lateral deviation** against the
same controller with learning off. Solver cost roughly doubles (4.5 → 9.4 ms mean,
p99 24.9 ms) — still inside a 50 ms budget.

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

## Open questions

- **The GP dictionary saturates at 8 points at 20 Hz**, against 199 (the `max_num_data_points:
  200` cap) when the same run is stepped at 50 Hz. `OnlineLearningStrategy`'s novelty
  criterion is presumably rejecting almost everything once the trajectory repeats, but 8 is
  low enough to be worth confirming rather than assuming. The learning benefit above is
  measured *with* only those 8 points, so the ceiling may be higher.
- **LBMPC computes its residuals against a zero input on the car.**
  `lb_mpc.py` sets `_prev_input_for_residual = np.zeros(2)` with `self.u0.copy()` commented
  out beside it, so `r = x_k − f(x_{k−1}, 0)` attributes the whole effect of the applied
  input to model error. race_stack `aeb5101d9` (2026-02-22, "Improvement in residual
  calculation validated") — pre-existing, and the sim adapter passes the real `u0` instead.
  Worth a decision before the next car run.
- `StateIndex.INPUT_JERK` / `INPUT_DSTEERING` should be restored in
  `forzaeth_mpc/utils/indices.py` (and race_stack's copy); they were dropped by a merge
  revert, `94ea48a79`.
