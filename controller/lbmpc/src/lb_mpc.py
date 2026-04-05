import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Union

import gpytorch
import numpy as np
import torch
from frenet_converter.frenet_converter import FrenetConverter
from l4acados.controllers import ResidualLearningMPC, ZeroOrderGPMPC
from l4acados.models import GPyTorchResidualModel, PyTorchFeatureSelector
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_data_processing_strategy import (
    OnlineLearningStrategy, RecordDataStrategy, VoidDataStrategy)
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel, BatchIndependentMultitaskGPModel)
from lbmpc.src.acados_solver_settings import build_ocp_for_l4acados
from lbmpc.src.utils.indicies import InputIndex, StateIndex
from lbmpc.src.utils.utils import check_file_changes, load_tensor_data
from pbl_config import (CarConfig, LBMPCConfig, PacejkaTireConfig,
                        TrailingConfig)
from scipy.integrate import solve_ivp
from single_track_mpc.utils.splinify import SplineTrack

from .offline_viz_logger import OfflineMPCLogger

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None


class LBMPC:
    def __init__(self,
                 pose_frenet: np.ndarray,
                 waypoints_data: np.ndarray,
                 racecar_version: str,
                 lbmpc_config: LBMPCConfig,
                 car_config: CarConfig,
                 tire_config: PacejkaTireConfig,
                 trailing_config: TrailingConfig,
                 controller_frequency: float,
                 using_gokart: bool,
                 lbmpc_path: str,
                 params_path: str) -> None:

        # Store configs
        self.racecar_version = racecar_version
        self.lbmpc_config = lbmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # TODO: FORCE MODEL MISMATCH TO TEST GP
        # self.tire_config.friction_coeff = 0.5
        # self.car_config.m = 10

        # Initialize parameters
        self._mpc_init_params()

        # Initialize solver with waypoints
        self._mpc_initialize_solver(pose_frenet, waypoints_data)

        # Initialize Residual model
        self.lbmpc_path = lbmpc_path
        self.use_gp = self.lbmpc_config.gp_params.use_gp

        # Build the B matrix, B_d_pinv and nw
        idx = np.asarray(self.lbmpc_config.gp_params.unc_jac_G_mat_idx, dtype=float)
        self.B = np.diag(idx)
        self.B = self.B[:, ~np.all(self.B == 0, axis=0)]
        self.B_d_pinv = np.linalg.pinv(self.B)
        self._nw_gp = self.B.shape[1]

        if self.use_gp:
            self._init_residual_model()
        else:
            self.residual_gp = None
            self.use_inducing_gp = False
            self.learn_online = False
            self.B = None
            self._viz_simple = None

        solver_settings_file_name = os.path.join(self.lbmpc_path, "src", "acados_solver_settings.py")
        bicycle_model_file_name = os.path.join(self.lbmpc_path, "src", "bicycle_model.py")
        params_file_name = os.path.join(params_path, "lb_mpc_params.yaml")
        mpc_files_to_watch = [
            solver_settings_file_name,
            bicycle_model_file_name,
            params_file_name,
        ]
        build_hash_path = os.path.join(self.lbmpc_path, "data/hash", "build_c_code_hash.md5")
        path_json_ocp = os.path.join(self.lbmpc_path, "data/ocp", "lbmpc_ocp_solver_config.json")
        path_json_sim = os.path.join(self.lbmpc_path, "data/ocp", "lbmpc_sim_solver_config.json")
        build_c_code = check_file_changes(mpc_files_to_watch, build_hash_path)
        build_c_code = build_c_code or not self._cython_solver_artifacts_available(
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
        )

        # Create the Residual model from l4acados
        self.l4a_solver = ZeroOrderGPMPC(
            ocp=self.ocp,
            B=self.B,
            residual_model=self.residual_gp,
            use_cython=True,
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
            build_c_code=build_c_code,
        )

        # ---- offline viz logger ----
        try:
            nx = int(self.ocp.dims.nx)
            nu = int(self.ocp.dims.nu)
            N = int(self.lbmpc_config.N)
            state_names = [e.name.lower() for e in sorted(StateIndex, key=lambda x: x.value)]
            input_names = [e.name.lower() for e in sorted(InputIndex, key=lambda x: x.value)]

            self.offlog = OfflineMPCLogger(
                root_dir=self.lbmpc_path,
                nx=nx, nu=nu, N=N,
                state_names=state_names,
                input_names=input_names,
                enable=True,
                flush_every=25,
                horizon_stride=4,
            )
        except Exception as e:
            print(f"[offline_viz] logger disabled: {e}")
            self.offlog = None

        # Keep a handle to the sim solver and set its default horizon
        if hasattr(self.l4a_solver, "sim_solver"):
            self.sim_solver = self.l4a_solver.sim_solver
            # Typically Ts is in ocp.solver_options.Tsim
            self.sim_solver.set("T", float(self.ocp.solver_options.Tsim))
        else:
            self.sim_solver = None

        # Apply warm start
        self._apply_warm_start(pose_frenet=pose_frenet)

        self._prev_state_for_residual = None
        self._prev_input_for_residual = None
        self.last_status = 0
        self.x_pred_nom = None
        self.pred_x = None
        self.previous_pred_x = None
        self.previous_pred_u = None
        self.count = 0

    @staticmethod
    def _cython_solver_artifacts_available(path_json_ocp: str, path_json_sim: str) -> bool:
        for json_path, module_stem in (
            (path_json_ocp, "acados_ocp_solver_pyx"),
            (path_json_sim, "acados_sim_solver_pyx"),
        ):
            code_export_directory = LBMPC._read_code_export_directory(json_path)
            if not code_export_directory:
                return False

            artifact_pattern = os.path.join(code_export_directory, f"{module_stem}*.so")
            if not glob(artifact_pattern):
                return False

        return True

    @staticmethod
    def _read_code_export_directory(json_path: str) -> Union[str, None]:
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as json_file:
                match = re.search(
                    r'"code_export_directory"\s*:\s*"([^"]+)"',
                    json_file.read(),
                )
        except OSError:
            return None

        if not match:
            return None

        return os.path.abspath(match.group(1))

    def _mpc_init_params(self) -> None:
        # MPC parameters
        self.sampling_time_mpc = 1 / self.lbmpc_config.MPC_freq
        self.comput_time = 0.01  # Initial guess, updated with measured computation times
        self.t_delay = self.lbmpc_config.t_delay # + self.comput_time
        self.ma_alpha = self.sampling_time_mpc / (self.sampling_time_mpc + 5.0)  # Moving average window of 1.0 seconds
        # buffer to save last control inputs
        self.buffer_len = 20
        self.buffer_prev_solutions = np.zeros((2, self.buffer_len), dtype=np.float64)
        self.buf_filled = 0
        self.delay_sim = self.lbmpc_config.delay_sim

        # Steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.lbmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.overtake_d = self.lbmpc_config.overtake_d
        self.loop_rate = self.controller_frequency

        # Gap control variables
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

        # First time that exceeds certain velocity
        self.first_time_done = False
        self._gp_warmup_done = False

    def _mpc_initialize_solver(self, pose_frenet: np.ndarray, waypoints_data: np.ndarray) -> None:
        """Initialize solver with waypoints data (no ROS dependency)"""

        # Extract waypoint data - assuming format: [x, y, vx, kappa, s, d_left, d_right]
        x = waypoints_data[:, 0]
        y = waypoints_data[:, 1]
        # vx_ref = np.full_like(waypoints_data[:, 2], 10.0, dtype=float)  # waypoints_data[:, 2]
        kapparef = waypoints_data[:, 3]
        self.s_ref = waypoints_data[:, 4]
        d_left = waypoints_data[:, 5]
        d_right = waypoints_data[:, 6]

        # Initialize frenet converter
        self.fren_conv = FrenetConverter(x, y)

        coords_path = self._transform_waypoints_to_coords_from_array(waypoints_data=waypoints_data)
        self.spline = SplineTrack(coords_direct=coords_path)

        # Initialize lap counter
        self.nr_laps = 0

        # Setup acados solver
        (self.ocp, self.model, self.constraint, self.model_params) = build_ocp_for_l4acados(
            self.s_ref, kapparef, d_left, d_right,
            self.lbmpc_config, self.car_config, self.tire_config
        )

        self.set_sim_mu(self.tire_config.friction_coeff)

        self.kappa = kapparef
        self.prev_acc = 0

    def _init_residual_model(self) -> None:
        """
        Initialize GP residual model and data-processing strategy.
        Keeps LBMPC formulation intact; only adds a model
        that can learn residuals y = x(k) - x_pred(k) (or a subset).
        """

        gp_cfg = self.lbmpc_config.gp_params

        self.use_inducing_gp = gp_cfg.use_inducing_gp
        self.min_points_before_using_gp = gp_cfg.min_points_before_using_gp
        self.max_num_data_points = gp_cfg.max_num_data_points

        self._nx_gp = int(self.model.n_x)
        self._nu_gp = int(self.model.n_u)

        # Feature selection over x
        input_selection = gp_cfg.input_feature_selection
        if len(input_selection) != (self._nx_gp + self._nu_gp):
            raise ValueError(
                f"gp_params['input_feature_selection'] must have length {self._nx_gp + self._nu_gp}"
            )
        self._input_feature_selector = PyTorchFeatureSelector(input_selection)

        z_path = os.path.join(self.lbmpc_path, "data/datapoints", "z_data.csv")
        y_path = os.path.join(self.lbmpc_path, "data/datapoints", "y_data.csv")

        # Data strategy
        self.mode = gp_cfg.data_processing_mode
        if self.mode == "none" or self.mode == "use_prerecorded_data":
            data_processing_strategy = VoidDataStrategy()
        elif self.mode == "record":
            data_processing_strategy = RecordDataStrategy(z_path, y_path)
        elif self.mode == "learn":
            data_processing_strategy = OnlineLearningStrategy(self.max_num_data_points)
        else:
            raise ValueError(
                f"Invalid data_processing_mode: {self.mode}. Must be one of ['none', 'record', 'learn']"
            )

        # buffers to save first datapoints until min_points_before_using_gp is reached
        self._gp_buffer_z = []
        self._gp_buffer_y = []

        # In case we want to try an already learned model in another track
        z_train_data, y_train_data = None, None
        if self.mode == "use_prerecorded_data":
            z_train_data, y_train_data = load_tensor_data(z_path, y_path)

            # Apply feature selectionº
            z_train_data = self._input_feature_selector(z_train_data)

            if z_train_data.shape[0] > self.max_num_data_points:
                z_train_data = z_train_data[:self.max_num_data_points, :]
                y_train_data = y_train_data[:self.max_num_data_points, :]

        # Create one fake datapoint because l4acados requires it
        # if self.mode == "learn":
        #     z_train_data = torch.zeros((1, sum(input_selection)), dtype=torch.float32)
        #     y_train_data = torch.zeros((1, self._nw_gp), dtype=torch.float32)

        # Build GP model: full or inducing
        if not self.use_inducing_gp:
            gp_model = BatchIndependentMultitaskGPModel(
                train_x=z_train_data,
                train_y=y_train_data,
                likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self._nw_gp,
                ),
                residual_dimension=self._nw_gp,
                input_dimension=sum(input_selection),
                use_ard=True,
            )
            hyperparameter_names_dict = {
                "likelihood.task_noises": "likelihood.task_noises",
                "kernel.lengthscale": "covar_module.base_kernel.lengthscale",
                "kernel.outputscale": "covar_module.outputscale",
            }
        else:
            num_inducing = gp_cfg.num_inducing_points
            in_dim = int(sum(input_selection))
            inducing_zeros = torch.zeros((num_inducing, in_dim), dtype=torch.float32)
            gp_model = BatchIndependentInducingPointGPModel(
                train_x=z_train_data,
                train_y=y_train_data,
                likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self._nw_gp,
                ),
                residual_dimension=self._nw_gp,
                input_dimension=sum(input_selection),
                inducing_points=inducing_zeros,
                use_ard=True,
            )
            hyperparameter_names_dict = {
                "likelihood.task_noises": "likelihood.task_noises",
                "kernel.lengthscale": "covar_module.base_kernel.base_kernel.lengthscale",
                "kernel.outputscale": "covar_module.base_kernel.outputscale",
            }

        # Initialize hyperparameters if provided
        hyper_cfg = gp_cfg.hyperparameters
        if self.mode in ("use_prerecorded_data", "learn") and hyper_cfg:
            gp_model.initialize(
                **{
                    hyperparameter_names_dict[key]: torch.Tensor(hyper_cfg[key])
                    for key in hyperparameter_names_dict.keys()
                    if key in hyper_cfg
                }
            )

        gp_model.eval()
        gp_model.likelihood.eval()

        if self.mode == "learn":
            gp_model.train_inputs = None
            gp_model.train_targets = None
            gp_model.prediction_strategy = None

        # Wrap in residual model
        self.residual_gp = GPyTorchResidualModel(
            gp_model=gp_model,
            feature_selector=self._input_feature_selector,
            data_processing_strategy=data_processing_strategy,
        )

    def main_loop(
            self,
            state: str,
            position_in_map: np.ndarray,
            waypoint_array_in_map: np.ndarray,
            speed_now: float,
            opponent: Union[np.ndarray, None],
            position_in_map_frenet: np.ndarray,
            alpha: float,
            learning_based_state: np.ndarray,
            track_length: float,
            compute_time: float):
        """
        Returns:
            speed, acceleration, jerk, steering_angle, states(list|None), status(int)
        """

        self.state = state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.alpha = alpha
        self.vel_y = learning_based_state[0]
        self.yaw_rate = learning_based_state[1]
        if self.using_gokart:
            self.measured_acc = learning_based_state[2]
            self.measured_steer = learning_based_state[3]
        else:
            self.measured_steer = self.steering_angle_buf[-1]
            self.measured_acc = getattr(self, "prev_acc", 0.0)
        self.track_length = self.spline.track_length
        self.comp_time = compute_time

        _t0 = time.perf_counter()
        init_time = 0.0
        residual_time = 0.0
        rest_time = 0.0

        # Heading correction
        center_car_s = self.position_in_map_frenet[0] + self.car_config.lr * np.cos(self.alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = self.position_in_map[0, 2] - np.arctan2(deriv_center[1], deriv_center[0])
        alpha_center = alpha_center % (2 * np.pi)
        if alpha_center > np.pi:
            alpha_center -= 2 * np.pi

        # Lap counter
        if self.position_in_map_frenet[0] < self.previous_frenet_s - 1:
            self.nr_laps += 1
        self.previous_frenet_s = self.position_in_map_frenet[0]
        current_pos_s = self.position_in_map_frenet[0] + self.nr_laps * self.track_length

        self.fre_s = current_pos_s
        self.fre_d = self.position_in_map_frenet[1]
        self.fre_alpha = alpha_center

        x0 = np.array([
            self.fre_s, self.fre_d, self.fre_alpha,
            speed_now, self.vel_y, self.yaw_rate, self.measured_steer, self.measured_acc
        ], dtype=np.float64)

        _t1 = time.perf_counter()
        init_time = _t1 - _t0

        if speed_now > 3.0 and not self.first_time_done:
            self.first_time_done = True
            print("First time done!")

        # Simulated delay compensation
        self.propagated_x = self._propagate_time_delay(x0) if self.delay_sim else x0

        # Record new datapoints. Until we don't have at least self.min_points_before_using_gp
        # if (
        #     self.use_gp
        #     and self.residual_gp is not None
        #     and getattr(self, "mode", None) in ("learn", "record")
        # ):
        #     res = self._compute_residual(new_state=x0)
        #     if res is not None:
        #         z_train, residual = res

        #         # 1) If warm-up is not finished, collect into buffer
        #         if not self._gp_warmup_done:
        #             self._gp_buffer_z.append(z_train)
        #             self._gp_buffer_y.append(residual)

        #             # When we reach the threshold, flush buffer into the GP
        #             if len(self._gp_buffer_z) >= self.min_points_before_using_gp:
        #                 for z_b, y_b in zip(self._gp_buffer_z, self._gp_buffer_y):
        #                     self.residual_gp.record_datapoint(z_b, y_b)

        #                 # Clear buffer and mark GP as ready
        #                 self._gp_buffer_z.clear()
        #                 self._gp_buffer_y.clear()
        #                 self._gp_warmup_done = True
        #                 print("GP warmup done")

        #         # 2) If warm-up is finished, directly add each new datapoint
        #         else:
        #             self.residual_gp.record_datapoint(z_train, residual)
        res = None
        measured_residual = None
        last_pred_residual = None
        self.nominal_pred_x = None
        self.nomres_pred_x = None
        if (
            self.use_gp
            and self.residual_gp is not None
            and getattr(self, "mode", None) in ("learn", "record")
        ):
            t_res_start = time.perf_counter()
            res = self._compute_residual(new_state=x0)
            self.count += 1
            if res is not None and (self.last_status == 0 or self.last_status ==
                                    2) and self.nr_laps >= 2 and self.count >= 1:
                z_train, residual = res
                self.residual_gp.record_datapoint(z_train, residual)
                self.count = 0
                self._gp_warmup_done = True
            residual_time += time.perf_counter() - t_res_start

        if self.pred_x is not None and res is not None:
            # measured residual r_k = B^+ (x_k - f_nominal(x_{k-1}, u_{k-1}))
            _, measured_residual = res

            # Predicted residual {mpc first solution - nominal(x_prev, u_prev)}
            nominal_pred_x = self._predict_next_state_nominal(
                self._prev_state_for_residual,
                self._prev_input_for_residual)  # (self.previous_pred_x, self.previous_pred_u)
            nomres_pred_x = self._predict_next_state_with_gp(self._prev_state_for_residual,
                                                             self._prev_input_for_residual)
            last_pred_residual = self.B_d_pinv @ (nomres_pred_x - nominal_pred_x)
            self.nominal_pred_x = nominal_pred_x
            self.nomres_pred_x = nomres_pred_x

        if (measured_residual is not None) and (last_pred_residual is not None):
            mr = np.asarray(measured_residual, dtype=np.float64).reshape(-1)
            pr = np.asarray(last_pred_residual, dtype=np.float64).reshape(-1)

            residual_dim = min(mr.size, pr.size, self._res_abs_err_sum.size)
            if residual_dim > 0:
                abs_err = np.abs(pr[:residual_dim] - mr[:residual_dim])
                self._res_abs_err_sum[:residual_dim] += abs_err
                self._res_err_count += 1

        viz_information = None
        if (x0 is not None) and (self.nomres_pred_x is not None):
            # If nominal_pred_x was not computed this step, fill with NaNs to keep size consistent
            if self.nominal_pred_x is None:
                nominal_pred_x_for_viz = np.full_like(self.nomres_pred_x, np.nan, dtype=np.float64)
            else:
                nominal_pred_x_for_viz = np.asarray(self.nominal_pred_x, dtype=np.float64).reshape(-1)

            state_for_viz = np.asarray(x0, dtype=np.float64).reshape(-1)
            pred_for_viz = np.asarray(self.nomres_pred_x, dtype=np.float64).reshape(-1)

            # Visualize the real one step mismatch without propagating delay
            viz_information = np.concatenate(
                [state_for_viz, pred_for_viz, nominal_pred_x_for_viz]
            )

        rest_start = time.perf_counter()

        # Update inducing points
        self._update_inducing_points_along_horizon()

        # Fix initial state
        self.l4a_solver.set(0, "lbx", self.propagated_x)
        self.l4a_solver.set(0, "ubx", self.propagated_x)

        # Trailing control
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self._trailing_controller(self.waypoint_array_in_map[0, 2])
        else:
            self.i_gap = 0  # reset integral if not trailing

        for i in range(self.lbmpc_config.N + 1):
            if (self.state == "TRAILING") and (self.opponent is not None):
                target_v_speed = self.trailing_command
            else:
                idx = np.abs(self.waypoint_array_in_map[:, 4] - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                target_v_speed = self.waypoint_array_in_map[idx, 2]
                self.overtake_d = 0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            if i < self.lbmpc_config.steps_delay:
                multiplier = 1e6
                online_parameters = np.array([
                    target_v_speed,
                    self.lbmpc_config.qadv,
                    self.lbmpc_config.qv,
                    self.lbmpc_config.qn,
                    self.lbmpc_config.qalpha,
                    self.lbmpc_config.qjerk,
                    multiplier * self.lbmpc_config.qddelta,
                    self.lbmpc_config.alat_max,
                    self.lbmpc_config.a_min,
                    self.lbmpc_config.a_max,
                    self.lbmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.lbmpc_config.terminal_multiplier
                ], dtype=np.float64)
            else:
                online_parameters = np.array([
                    target_v_speed,
                    self.lbmpc_config.qadv,
                    self.lbmpc_config.qv,
                    self.lbmpc_config.qn,
                    self.lbmpc_config.qalpha,
                    self.lbmpc_config.qjerk,
                    self.lbmpc_config.qddelta,
                    self.lbmpc_config.alat_max,
                    self.lbmpc_config.a_min,
                    self.lbmpc_config.a_max,
                    self.lbmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.lbmpc_config.terminal_multiplier
                ], dtype=np.float64)

            self.l4a_solver.set(i, "p", online_parameters)

            if i < self.lbmpc_config.N:
                self.l4a_solver.set(i, "lbu", np.array([self.lbmpc_config.ddelta_min, -50]))
                self.l4a_solver.set(i, "ubu", np.array([self.lbmpc_config.ddelta_max, 50]))
                self.l4a_solver.set(i + 1, "lbx",
                                    np.array([self.lbmpc_config.v_min,
                                              self.lbmpc_config.delta_min,
                                              self.lbmpc_config.a_min]))
                self.l4a_solver.set(i + 1, "ubx",
                                    np.array([self.lbmpc_config.v_max,
                                              self.lbmpc_config.delta_max,
                                              self.lbmpc_config.a_max]))

        # Solve OCP
        try:
            status = self.l4a_solver.solve()
        except Exception as e:
            msg = str(e)
            m = re.search(r'status\s+(\d+)', msg)
            status = int(m.group(1)) if m else -1
            print(
                f"[MPC Controller] Solver failed with message={msg}.\n Status={status}.\n Applying warm start again.")
            self._apply_warm_start(
                pose_frenet=[
                    self.fre_s,
                    self.fre_d,
                    self.fre_alpha],
                actual_speed=self.speed_now,
                actual_acc=self.measured_acc)

        self.last_status = status

        # Actual cost of the problem, in case wanting to log
        # current_cost = self.l4a_solver.get_cost()

        self.u0 = self.l4a_solver.get(0, "u")
        self.prev_acc = self.l4a_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.lbmpc_config.steps_delay + 1  # if not self.delay_sim else 1
        # delayed_index = delayed_index if speed_now > 2.0 else delayed_index + 3
        self.pred_x = self.l4a_solver.get(delayed_index, "x")

        self.steering_angle = self.pred_x[StateIndex.STEERING_ANGLE_DELTA.value]
        self.speed = self.pred_x[StateIndex.VELOCITY_V_X.value]
        self.acceleration = self.prev_acc + self.u0[InputIndex.JERK.value] / self.controller_frequency
        self.prev_acc = self.acceleration
        self.jerk = self.u0[InputIndex.JERK.value]

        # Metrics calculation purposes
        self.previous_pred_x = self.l4a_solver.get(delayed_index - 1, "x")
        self.previous_pred_u = self.l4a_solver.get(delayed_index - 1, "u")

        # Update steering buffer
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.steering_angle

        self.mpc_sd = np.array([
            self.l4a_solver.get(j, "x")[:2] for j in range(self.lbmpc_config.N + 1)
        ])

        self._prev_state_for_residual = x0.copy()
        self._prev_input_for_residual = np.zeros(2) # self.u0.copy()    # [ddelta, jerk]

        # # Comput time
        # # Comput time (total + per-section)
        # _now = time.perf_counter()
        # rest_time = _now - rest_start          # everything after rest_start
        # _elapsed = _now - _t0                  # full main_loop duration
        _elapsed = 0.0

        # Update computational time calculation with moving average
        self.comput_time = self.comput_time * self.ma_alpha + _elapsed * (1 - self.ma_alpha)
        if self._gp_warmup_done:
            self.t_delay = float(self.lbmpc_config.t_delay) + 0.01
        else:
            self.t_delay = float(self.lbmpc_config.t_delay) # + float(self.comput_time)

        # Shift all solutions and att the new one
        self.buffer_prev_solutions[:, 1:] = self.buffer_prev_solutions[:, :-1]
        self.buffer_prev_solutions[:, 0] = np.asarray(self.u0, dtype=np.float64).reshape(2,)
        self.buf_filled = min(self.buf_filled + 1, self.buffer_len)

        # Comput time
        # Comput time (total + per-section)
        _now = time.perf_counter()
        rest_time = _now - rest_start          # everything after rest_start
        _elapsed = _now - _t0                  # full main_loop duration

        if not hasattr(self, "_timing_sum"):
            self._timing_sum = 0.0
            self._timing_count = 0
            self._timing_max = 0.0
            self._timing_last_report = time.perf_counter()
            self._timing_init_sum = 0.0
            self._timing_residual_sum = 0.0
            self._timing_rest_sum = 0.0

        # --- Residual performance accumulators (lazy init) ---
        if not hasattr(self, "_res_abs_err_sum"):
            self._res_abs_err_sum = np.zeros(self._nw_gp, dtype=np.float64)
            self._res_err_count = 0
            self._res_last_lap = -1

        self._timing_sum += _elapsed
        self._timing_init_sum += init_time
        self._timing_residual_sum += residual_time
        self._timing_rest_sum += rest_time
        self._timing_count += 1

        if _elapsed > self._timing_max:
            self._timing_max = _elapsed

        # Print once per lap
        if self.nr_laps > getattr(self, "_timing_last_lap", -1):
            denom = max(1, self._timing_count)

            avg_ms = (self._timing_sum / denom) * 1e3
            max_ms = self._timing_max * 1e3
            init_ms = (self._timing_init_sum / denom) * 1e3
            residual_ms = (self._timing_residual_sum / denom) * 1e3
            rest_ms = (self._timing_rest_sum / denom) * 1e3

            print(
                f"[LBMPC] Lap {self.nr_laps}: "
                f"avg = {avg_ms:.2f} ms "
                f"(init={init_ms:.2f}, residual={residual_ms:.2f}, rest={rest_ms:.2f}) "
                f"max = {max_ms:.2f} ms over {self._timing_count} iters"
            )

            if self._res_err_count > 0:
                mae = self._res_abs_err_sum / float(self._res_err_count)
                residual_terms = ", ".join(
                    f"r[{idx}]={value:.4e}" for idx, value in enumerate(mae)
                )
                print(f"[Residual] Lap {self.nr_laps}: MAE | {residual_terms} (based on {self._res_err_count} samples)")
            else:
                print(f"[Residual] Lap {self.nr_laps}: no residual data available this lap.")

            # Reset residual counters for the next lap
            self._res_abs_err_sum[:] = 0.0
            self._res_err_count = 0
            self._res_last_lap = self.nr_laps
            # ---------------------------------------------

            # reset timing stats for the next lap
            self._timing_sum = 0.0
            self._timing_init_sum = 0.0
            self._timing_residual_sum = 0.0
            self._timing_rest_sum = 0.0
            self._timing_count = 0
            self._timing_max = 0.0
            self._timing_last_lap = self.nr_laps

        # ---- offline logging (best-effort, never break control loop) ----
        if getattr(self, "offlog", None) is not None:
            try:
                # gp size best-effort
                gp_n = None
                try:
                    gp_n = int(self.residual_gp.gp_model.train_inputs[0].shape[0])
                except Exception:
                    gp_n = None

                x_nom = self._predict_next_state_nominal(self.previous_pred_x, self.previous_pred_u)
                self.offlog.log_step(
                    lap=self.nr_laps,
                    status=self.last_status,
                    x_meas=x0,  # Actual measured state, corresponding to time td
                    x_nom_delayed=self.nominal_pred_x,  # Nominal prediction corresponding to time td
                    x_pred_delayed=self.nomres_pred_x,  # Total prediction corresponding to time td
                    x_meas_prop=self.propagated_x,  # Propagated measured state corresponding to the actual t = 0
                    x_pred=self.pred_x,  # Propagated measure
                    x_nom=x_nom,
                    u0=self.u0,
                    t_total=_elapsed, t_init=init_time, t_residual=residual_time, t_rest=rest_time,
                    r_meas=measured_residual,
                    r_pred=last_pred_residual,
                    gp_num_points=gp_n
                )

                # horizon logging: predicted traj + bounds + covariances
                # cov_arr = getattr(self.l4a_solver, "covariances_array", None)
                # # indices used by ZoRO tightening (keep them in your config and pass here)
                # idx_lh_t = getattr(self.ocp.zoro_description, "idx_lh_t", [])
                # idx_uh_t = getattr(self.ocp.zoro_description, "idx_uh_t", [])
                # self.offlog.log_horizon(
                #     solver=self.l4a_solver,
                #     lap=self.nr_laps,
                #     covariances_array=cov_arr,
                #     idx_lh_t=list(idx_lh_t) if idx_lh_t is not None else [],
                #     idx_uh_t=list(idx_uh_t) if idx_uh_t is not None else [],
                # )

                self.offlog.tick()
            except Exception as e:
                print(f"[offline_viz] log failed: {e}")

        states_list = None
        if status == 0 or status == 2:
            states_list = []
            for node in range(self.lbmpc_config.N + 1):
                result = self.l4a_solver.get(node, "x").tolist()
                for item in result:
                    states_list.append(item)
            return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status, viz_information
        else:
            return 0.0, 0.0, 0.0, self.measured_steer, states_list, status, viz_information

    def _apply_warm_start(self, pose_frenet: np.ndarray, actual_speed=1.0, actual_acc=1.0):
        """
        Generates and applies a simple initial guess considering constant inputs.
        """
        nx = int(self.model.n_x)
        nu = int(self.model.n_u)
        N = self.lbmpc_config.N

        x0 = np.zeros(nx)
        x0[StateIndex.POS_ON_CENTER_LINE_S.value] = pose_frenet[0]
        x0[StateIndex.MIN_DIST_TO_CENTER_LINE_N.value] = pose_frenet[1]
        x0[StateIndex.ORIENTATION_THETA.value] = pose_frenet[2]
        x0[StateIndex.VELOCITY_V_X.value] = max(actual_speed, 1.0)
        x0[StateIndex.ACCEL.value] = actual_acc  # Maybe 0.0

        const_steer_vel = 0.0  # to figure out
        u0 = np.zeros(nu)
        u0[InputIndex.D_STEERING_ANGLE.value] = const_steer_vel

        warm_start = np.zeros((N + 1, nx))
        warm_start[0, :] = x0
        x_prev = warm_start[0, :]
        for i in range(1, N + 1):
            # TODO: do it with residual _predict_next_state_with_gp
            x_next = self._predict_next_state_nominal(x_prev, u0)
            warm_start[i, :] = x_next
            self.l4a_solver.set(i, "x", x_next)
            if i < N:
                self.l4a_solver.set(i, "u", u0)
            x_prev = x_next

    # TODO: Shouldn't we use the previous results?
    def _update_warm_start(self) -> None:
        """
        Propagate states along the horizon with the current inputs,
        using the learned model (nominal + GP).
        """
        for i in range(self.lbmpc_config.N + 1):
            current_state = self.l4a_solver.get(i, "x")
            if i < self.lbmpc_config.N:
                current_input = self.l4a_solver.get(i, "u")
            else:
                current_input = np.zeros(self.model.n_u)

            # TODO: Use also the residual
            # updated_state = self._predict_next_state_with_gp(current_state, current_input)
            updated_state = self._predict_next_state_nominal(current_state, current_input)
            self.l4a_solver.set(i, "x", updated_state)

            if i < self.lbmpc_config.N - 1:
                next_inputs = self.l4a_solver.get(i + 1, "u")
                self.l4a_solver.set(i, "u", next_inputs)

    def _propagate_time_delay(self, x0: np.ndarray) -> np.ndarray:
        """
        Forward propagate current state over self.t_delay with discrete steps of the nominal model.
        """
        x = x0
        n_steps = int(np.floor(self.t_delay / self.sampling_time_mpc))
        n_steps = min(n_steps, self.buffer_len, self.buf_filled)
        if x[StateIndex.VELOCITY_V_X.value] > 1.5:
            for j in range(n_steps - 1, -1, -1):
                x = self._predict_next_state_nominal( # _predict_next_state_with_gp(
                    x,
                    self.buffer_prev_solutions[:, j]
                )

        # Saturations as before
        if abs(x[StateIndex.STEERING_ANGLE_DELTA.value]) > self.constraint.delta_max:
            x[StateIndex.STEERING_ANGLE_DELTA.value] = np.sign(
                x[StateIndex.STEERING_ANGLE_DELTA.value]) * self.constraint.delta_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) > self.constraint.v_x_max:
            x[StateIndex.VELOCITY_V_X.value] = np.sign(x[StateIndex.VELOCITY_V_X.value]) * self.constraint.v_x_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) < self.constraint.v_x_min:
            x[StateIndex.VELOCITY_V_X.value] = self.constraint.v_x_min

        return x

    def _compute_residual(self, new_state: np.ndarray):
        """
        Compute residual y = B^+ (x_meas(k) - f(x(k-1), u(k-1))).
        new_state must be the measured state at time k (same dimension as model state).
        Returns (x_train, residual) or None.
        """
        if self._prev_state_for_residual is None or self._prev_input_for_residual is None:
            return None

        x_prev = self._prev_state_for_residual
        u_prev = self._prev_input_for_residual

        # 1-step prediction with nominal model
        x_pred_nom = self._predict_next_state_nominal(x_prev, u_prev)

        # Training input: [x_prev, u_prev] (full vector)
        z_train = np.concatenate((x_prev, u_prev))

        # Residual in selected coordinates
        residual = self.B_d_pinv @ (new_state - x_pred_nom)

        return z_train, residual

    def _trailing_controller(self, global_speed: float) -> float:
        # opponent: [s, d, v_s, v_d, has_los]
        self.gap = (self.opponent[0] - self.position_in_map_frenet[0]) % self.track_length
        self.gap_actual = self.gap

        if self.trailing_config.trailing_mode:
            self.gap_should = self.trailing_config.trailing_gap
        else:
            self.gap_should = self.position_in_map_frenet[2] * self.trailing_config.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff = self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error / self.loop_rate, -10.0, 10.0)

        p_value = self.gap_error * self.trailing_config.trailing_p_gain
        d_value = self.v_diff * self.trailing_config.trailing_d_gain
        i_value = self.i_gap * self.trailing_config.trailing_i_gain

        trailing_cmd = np.clip(self.opponent[2] - p_value - i_value - d_value, 0.0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            trailing_cmd = max(self.trailing_config.blind_trailing_speed, trailing_cmd)

        return trailing_cmd

    def _transform_waypoints_to_coords_from_array(self, waypoints_data: np.ndarray) -> np.ndarray:
        """
        Build track coordinates (left boundary, centerline, right boundary) from an
        ndarray of waypoints.

        Parameters
        ----------
        waypoints_data : np.ndarray
            Shape (N, >= 7) with columns:
            [x, y, vx, kappa, s, d_left, d_right]

        Returns
        -------
        res_coords : np.ndarray
            Shape (3, N, 2), stacked as:
            [boundaries_left, centerline, boundaries_right]
            where each slice is (N, 2) with XY points.
        """
        if waypoints_data.ndim != 2 or waypoints_data.shape[1] < 7:
            raise ValueError(
                "waypoints_data must be (N, >=7): [x, y, vx, kappa, s, d_left, d_right]"
            )

        # Extract columns we need
        xy = waypoints_data[:, :2].astype(float)            # (N, 2)
        d_left = waypoints_data[:, 5].astype(float).copy()  # (N,)
        d_right = waypoints_data[:, 6].astype(float).copy()  # (N,)

        # If the track is closed (last == first), drop the duplicated last point
        if np.allclose(xy[0], xy[-1], atol=1e-9):
            xy = xy[:-1]
            d_left = d_left[:-1]
            d_right = d_right[:-1]

        N = xy.shape[0]
        if N == 0:
            raise ValueError("Empty waypoint array after processing.")
        if N == 1:
            # Degenerate: just replicate centerline to both sides
            boundaries_left = xy.copy()
            boundaries_right = xy.copy()
            return np.stack([boundaries_left, xy, boundaries_right], axis=0)

        # Forward segment directions to next point
        next_xy = np.roll(xy, -1, axis=0)
        seg = next_xy - xy  # (N, 2)

        # If open track, the last segment should not wrap to the first
        open_track = not np.allclose(xy[0], xy[-1], atol=1e-9)
        if open_track:
            seg[-1] = xy[-1] - xy[-2]

        # Normalize robustly; fill invalid with previous valid or [1, 0]
        norms = np.linalg.norm(seg, axis=1)
        eps = 1e-12
        dir_xy = np.zeros_like(seg)
        valid = norms > eps
        dir_xy[valid] = (seg[valid].T / norms[valid]).T

        for i in range(N):
            if not valid[i]:
                dir_xy[i] = dir_xy[i - 1] if i > 0 else np.array([1.0, 0.0])

        # Left normal (+90° rotation): [-dy, dx]
        nL = np.column_stack((-dir_xy[:, 1], dir_xy[:, 0]))  # (N, 2)

        # Boundaries
        boundaries_left = xy + nL * d_left[:, None]
        boundaries_right = xy - nL * d_right[:, None]

        # Stack as required by SplineTrack(coords_direct=...)
        res_coords = np.stack([boundaries_left, xy, boundaries_right], axis=0)  # (3, N, 2)
        return res_coords

    MU_IDX = 12  # keep in class if you prefer

    def set_sim_mu(self, mu: float) -> None:
        """
        Update ONLY the simulator's parameter vector used in _dynamics_of_car().
        Does NOT touch the MPC solver's stage parameters.
        """
        # get current p as a flat numpy array
        p = np.array(self.model_params.p, dtype=float).reshape(-1).copy()
        p[self.MU_IDX] = float(mu)

        # write it back in the same type the model expects
        if _HAS_DM and isinstance(self.model_params.p, DM):
            self.model_params.p = DM(p)
        else:
            self.model_params.p = p

    def _update_inducing_points_along_horizon(self) -> None:
        """
        Update inducing point locations along the current open-loop MPC trajectory.

        Should be called once per MPC solve if using an inducing-point GP.
        """
        if not (self.use_gp and self.use_inducing_gp and self.residual_gp is not None):
            return

        gp_model = self.residual_gp.gp_model
        if not hasattr(gp_model, "num_inducing_points"):
            return

        # Choose evenly spaced indices along horizon
        inducing_indices = np.linspace(
            0,
            self.lbmpc_config.N,
            gp_model.num_inducing_points,
            dtype=int,
            endpoint=False,
        )

        states = np.array([self.l4a_solver.get(i, "x") for i in inducing_indices])
        inputs = np.array([self.l4a_solver.get(i, "u") for i in inducing_indices])
        xu = np.hstack((states, inputs))  # (M, nx+nu)

        gp_model.train()
        gp_model.initialize(
            **{
                "covar_module.inducing_points": self._input_feature_selector(
                    torch.Tensor(xu)
                )
            }
        )
        gp_model.eval()

    def _predict_next_state_nominal(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One-step nominal prediction using either acados sim_solver (if available)
        or the continuous model with Euler discretization.
        """
        # Use acados sim solver created by ResidualLearningMPC
        self.sim_solver.set("x", x.astype(float))
        self.sim_solver.set("u", u.astype(float))
        self.sim_solver.set("p", np.array(self.model_params.p, dtype=float))
        status = self.sim_solver.solve()  # TODO: Use status to print if something is wrong at least
        x_next = self.sim_solver.get("x")
        return np.array(x_next, dtype=float).reshape(-1)

    def _predict_next_state_with_gp(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Learned prediction:
            x_next ≈ f_nominal(x,u) + B * μ_residual(x,u)
        Only active if use_gp and warm-up already finished.
        """
        x_next_nom = self._predict_next_state_nominal(x, u)

        # Make sure self._gp_warmup_done is actually being set to True somewhere in main_loop!
        if not (self.use_gp and self.residual_gp is not None and getattr(self, "_gp_warmup_done", False)):
            return x_next_nom

        # 1. Prepare the input as a float32 PyTorch tensor
        xu = np.concatenate((x, u))[None, :]  # shape (1, nx+nu)
        xu_tensor = torch.tensor(xu, dtype=torch.float32)

        # 2. Get the GP prediction safely without tracking gradients
        with torch.no_grad():
            # Apply your configured feature selector (from _init_residual_model)
            xu_selected = self._input_feature_selector(xu_tensor)

            # Query the underlying GPyTorch model
            gp_pred = self.residual_gp.gp_model(xu_selected)

            # Extract the mean of the prediction and convert back to numpy
            mu_residual = gp_pred.mean.detach().cpu().numpy().reshape(-1)  # (nw,)

        # 3. Project the residual back into the full state space
        delta_x = self.B @ mu_residual  # (nx,)
        return x_next_nom + delta_x
