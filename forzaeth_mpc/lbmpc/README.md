# LB-MPC

LB-MPC is the single-track MPC with a learnt correction. A Gaussian process learns the error
of the nominal model, and the controller adds that error to its prediction.

The nominal model predicts:

`x_{k+1} = f_nominal(x_k, u_k)`

LB-MPC predicts:

`x_{k+1} = f_nominal(x_k, u_k) + B * mu_GP(z_k)`

`z_k` is `[x_k, u_k]` after feature selection. `mu_GP` is the mean of the Gaussian process.
`B` selects the states that receive the correction.

The Gaussian process is not part of the CasADi model. l4acados adds the correction outside
the model. The model file therefore contains no residual term.

## How to run it

Start the controller with `ctrl_algo:=LBMPC`:

```bash
roslaunch stack_master time_trials.launch ctrl_algo:=LBMPC
```

The controller reads `<config root>/<CAR>/lb_mpc_params.yaml`. Off a car, the config root is
the fixture directory [`config/`](../../config). On a car, it is
`$(rospack find stack_master)/config`, because `pbl_config` sets `FORZAETH_MPC_CONFIG`.

The `gp_params` block holds the parameters below.

## Parameters

- `use_gp`: switches the correction on. Set it to `false` to get the nominal MPC.
- `data_processing_mode`: how the controller treats the Gaussian process.
  - `learn`: learn while driving, and overwrite the saved model.
  - `use_learnt_model`: load the saved model, and learn nothing more.
- `learning_enabled`: permits learning at run time, in `learn` mode.
  - Set it to `false` to prepare the Gaussian process but record no data.
  - Dynamic reconfigure can set it to `true` later.
- `input_feature_selection`: selects the entries of `[x, u]` that the Gaussian process reads.
  - The order is `[s, n, theta, v_x, v_y, yaw_rate, delta, accel, ddelta, jerk]`.
  - `1` selects an entry. `0` ignores it.
  - `v_x`, `v_y`, `yaw_rate`, `delta` and `accel` carry the most information.
- `unc_jac_G_mat_idx`: selects the states that receive the correction.
  - The order is `[s, n, theta, v_x, v_y, yaw_rate, delta, accel]`.
  - `[0, 0, 0, 0, 1, 1, 0, 0]` corrects `v_y` and `yaw_rate` only.
  - This list is positional. It changes meaning if the state order changes.
- `max_num_data_points`: the largest number of points that the Gaussian process keeps.
- `use_inducing_gp`: uses an inducing-point Gaussian process instead of the full one.
- `num_inducing_points`: applies only if `use_inducing_gp` is `true`.
- `stochastic_mode`: how the controller propagates uncertainty. See "ZoRO" below.
- `enable_offline_logging`: writes the CSV log.
- `log_offline_horizon`: adds the uncertainty and tightening data of the horizon to that log.
  Use it only with a stochastic mode.

## Residual learning

The controller measures the error of one prediction step:

`r_k = B^+ (x_k - f_nominal(x_{k-1}, u_{k-1}))`

The Gaussian process then learns `r_k` from the selected features of `[x, u]`.

`unc_jac_G_mat_idx` decides what the Gaussian process learns. `input_feature_selection`
decides which signals it learns from.

The hyperparameters have these effects:

- `likelihood.task_noises`: the noise on each corrected state. A large value tells the
  Gaussian process to trust the data less.
- `kernel.lengthscale`: how fast the error changes with each feature. A small value gives a
  local and sensitive fit. A large value gives a smooth fit.
- `kernel.outputscale`: the size of the error. A large value lets the Gaussian process
  predict a large difference from the nominal model.

The shape of each hyperparameter must agree with `input_feature_selection` and
`unc_jac_G_mat_idx`. The controller fails to load a model that disagrees.

A saved model records the state order that it learnt under. The loader rejects a model from a
different state order, because the number of features and corrected states does not change
when the order changes.

## ZoRO

ZoRO is the stochastic mode. It propagates the uncertainty along the horizon, and it tightens
the constraints by that amount.

The deterministic mode uses the mean of the Gaussian process. ZoRO uses the mean and the
uncertainty.

The four settings differ only in the feedback that the propagation assumes:

- `disabled`: no propagation. The controller uses the mean only.
- `open_loop`: propagation without feedback. Under a friction error, the car leaves the track.
- `lqr`: propagation with one fixed feedback gain. **This is the only stochastic mode that
  works.**
- `riccati`: propagation with a feedback gain from the Riccati equations and the MPC weights.
  **This mode does not work.** The gain is zero, and the controller behaves like `open_loop`.
  It fails at the same step, and by the same distance. `BENCHMARK.md` gives the measurements.

## Workflow

1. Set `use_gp: true`, `data_processing_mode: "learn"` and `stochastic_mode: "disabled"`.
2. Select few states in `unc_jac_G_mat_idx`.
3. Select few features in `input_feature_selection`.
4. Drive until the learnt model is good, then set
   `data_processing_mode: "use_learnt_model"`.
5. To add robustness against the uncertainty, set `stochastic_mode: "lqr"`.
