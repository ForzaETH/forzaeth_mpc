# LBMPC

Learning-Based MPC adds a GP residual model on top of the nominal single-track MPC.

The nominal model predicts:
`x_{k+1} = f_nominal(x_k, u_k)`

The learnt controller predicts:
`x_{k+1} = f_nominal(x_k, u_k) + B mu_GP(z_k)`

with `z_k = [x_k, u_k]` after feature selection, `mu_GP` the GP mean, and `B` selecting which state channels receive the residual.

## Use

Run the controller with the argument `ctrl_algo:=LBMPC`:

```bash
roslaunch stack_master time_trials.launch ctrl_algo:=LBMPC
```

The main config is:
[lb_mpc_params.yaml](/home/gil/F1TENTH/race_stack/stack_master/config/NUC6/lb_mpc_params.yaml)

The relevant block is `gp_params`.

## Important Parameters

- `use_gp`: enables residual learning. If `false`, the controller is nominal MPC only.
- `data_processing_mode`: GP usage mode.
  - `learn`: learn online and keep overwriting the latest saved model.
  - `use_learnt_model`: load the last saved model and use it without learning new points.
- `input_feature_selection`: selects which entries of `[x, u]` are used as GP input.
  - Current order is `[x, u] = [s, n, alpha, vx, vy, yaw_rate, delta, accel, ddelta, jerk]`.
  - `1` means “use this feature”, `0` means “ignore it”.
  - Usually `vx, vy, yaw_rate, delta, accel` are the most useful dynamic features.
- `unc_jac_G_mat_idx`: selects which state residual channels are learnt.
  - Current state order is `[s, n, alpha, vx, vy, yaw_rate, delta, accel]`.
  - Example: `[0, 0, 0, 0, 1, 1, 0, 0]` means the GP learns residuals only on `vy` and `yaw_rate`.
- `max_num_data_points`: maximum GP dataset size kept online.
- `use_inducing_gp`: use an inducing-point GP instead of the full GP.
- `num_inducing_points`: only used when `use_inducing_gp: true`.
- `stochastic_mode`: selects the uncertainty propagation mode.
  - `disabled`: deterministic residual MPC.
  - `open_loop`: ZoRO without ancillary feedback.
  - `lqr`: ZoRO with a fixed linear feedback gain.
  - `riccati`: ZoRO with time-varying Riccati feedback.
- `enable_offline_logging`: enables the CSV logger.
- `log_offline_horizon`: logs horizon uncertainty/tightening data. This only makes sense for stochastic modes.

## Residual Learning

The residual is computed from the one-step prediction error:

`r_k = B^+ (x_k - f_nominal(x_{k-1}, u_{k-1}))`

The GP learns `r_k` as a function of selected features from `[x, u]`.

In practice:
- we use `unc_jac_G_mat_idx` to decide what to learn,
- and `input_feature_selection` to decide from which signals to learn it.


Some intuition for GP hyperparameters:
- `likelihood.task_noises`: measurement/noise level on each learnt residual channel. Larger values mean “trust the data less”.
- `kernel.lengthscale`: how quickly the residual changes with each input feature. Small means very local/sensitive, large means smoother.
- `kernel.outputscale`: overall residual amplitude. Larger means the GP expects bigger deviations from the nominal model.

> Note: the hyperparameter shapes must be consistent with the active GP dimensions defined by `input_feature_selection` and `unc_jac_G_mat_idx`.

## ZoRO

ZoRO is the stochastic version. It propagates uncertainty along the horizon and tightens constraints accordingly.

The difference to deterministic residual MPC is:
- deterministic: uses only the GP mean,
- stochastic: uses GP mean + uncertainty propagation for constraint tightening.

The three stochastic options differ only in how the ancillary feedback is chosen:
- `open_loop`: no feedback in the uncertainty propagation.
- `lqr`: one fixed feedback gain is used everywhere.
- `riccati`: a time-varying gain is computed from Riccati equations using the MPC weights. This is usually the most principled option and the default stochastic choice.

## Typical Workflow

1. Start with `use_gp: true`, `data_processing_mode: "learn"`, `stochastic_mode: "disabled"`.
2. Pick a small set of residual outputs with `unc_jac_G_mat_idx`.
3. Pick a small set of informative inputs with `input_feature_selection`.
4. Once the learnt model behaves well, switch to `data_processing_mode: "use_learnt_model"`.
5. If you want robustness to GP uncertainty, switch `stochastic_mode` to `riccati`.
