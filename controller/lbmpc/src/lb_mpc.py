import hashlib
import os
import re
from pathlib import Path
from typing import Union

import gpytorch
import numpy as np
import torch
from frenet_converter.frenet_converter import FrenetConverter
from l4acados.controllers import ResidualLearningMPC
from l4acados.models import GPyTorchResidualModel, PyTorchFeatureSelector
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_data_processing_strategy import (
    OnlineLearningStrategy, RecordDataStrategy, VoidDataStrategy)
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel, BatchIndependentMultitaskGPModel)
from lbmpc.src.acados_solver_settings import build_ocp_for_l4acados
from lbmpc.src.utils.indicies import InputIndex, StateIndex
from pbl_config import (CarConfig, LBMPCConfig, PacejkaTireConfig,
                        TrailingConfig)
from scipy.integrate import solve_ivp
from single_track_mpc.utils.splinify import SplineTrack

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
                 using_gokart: bool) -> None:

        # Store configs
        self.racecar_version = racecar_version
        self.lbmpc_config = lbmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # Initialize parameters
        self.mpc_init_params()

        # Initialize solver with waypoints
        self.mpc_initialize_solver(pose_frenet, waypoints_data)

        # Initialize Residual model
        self.lbmpc_path = "/home/gil/catkin_ws/src/race_stack/controller/lbmpc"  # TODO: handle this
        self.use_gp = self.lbmpc_config.gp_params.use_gp
        if self.use_gp:
            self._init_residual_model()
        else:
            self.residual_gp = None
            self.use_inducing_gp = False
            self.learn_online = False
            self.B = None

        mpc_files_to_watch = [
            "lbmpc/src/acados_solver_settings.py",
            "lbmpc/src/bicycle_model.py",
            "lbmpc/src/solver.yaml",               # TODO: to modify
        ]
        build_hash_path = "lbmpc/build/build_c_code_hash.md5"

        # TODO: To be used
        build_c_code = True  # check_file_changes(mpc_files_to_watch, build_hash_path)

        path_json_ocp = os.path.join(self.lbmpc_path, "data/ocp", "lbmpc_ocp_solver_config.json")
        path_json_sim = os.path.join(self.lbmpc_path, "data/ocp", "lbmpc_sim_solver_config.json")

        # Create the Residual model from l4acados
        self.l4a_solver = ResidualLearningMPC(
            ocp=self.ocp,
            B=self.B,
            residual_model=self.residual_gp,
            # use_cython=True, TODO: understand how works and instalation in docker
            path_json_ocp=path_json_ocp,
            path_json_sim=path_json_sim,
            build_c_code=build_c_code,
        )

        # Keep a handle to the sim solver and set its default horizon
        if hasattr(self.l4a_solver, "sim_solver"):
            self.sim_solver = self.l4a_solver.sim_solver
            # Typically Ts is in ocp.solver_options.Tsim
            self.sim_solver.set("T", float(self.ocp.solver_options.Tsim))
        else:
            self.sim_solver = None

        # Apply warm start
        self.apply_warm_start(pose_frenet=pose_frenet)

    def mpc_init_params(self) -> None:
        # MPC parameters
        self.t_MPC = 1 / self.lbmpc_config.MPC_freq
        self.t_delay = self.lbmpc_config.t_delay + self.t_MPC  # TODO: decide what to do with delay

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

    def mpc_initialize_solver(self, pose_frenet: np.ndarray, waypoints_data: np.ndarray) -> None:
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

        coords_path = self.transform_waypoints_to_coords_from_array(waypoints_data=waypoints_data)
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

        # Build the B matrix, B_d_pinv and nw
        idx = np.asarray(gp_cfg.unc_jac_G_mat_idx, dtype=float)
        self.B = np.diag(idx)
        self.B = self.B[:, ~np.all(self.B == 0, axis=0)]
        self.B_d_pinv = np.linalg.pinv(self.B)
        self._nw_gp = self.B.shape[1]

        # Feature selection over [x, u]
        input_selection = gp_cfg.input_feature_selection
        if len(input_selection) != (self._nx_gp + self._nu_gp):
            raise ValueError(
                f"gp_params['input_feature_selection'] must have length {self._nx_gp + self._nu_gp}"
            )
        self._input_feature_selector = PyTorchFeatureSelector(input_selection)

        x_path = os.path.join(self.lbmpc_path, "data/datapoints", "x_data.csv")
        y_path = os.path.join(self.lbmpc_path, "data/datapoints", "y_data.csv")

        # Data strategy
        self.mode = gp_cfg.data_processing_mode
        if self.mode == "none" or self.mode == "use_prerecorded_data":
            data_processing_strategy = VoidDataStrategy()
        elif self.mode == "record":
            data_processing_strategy = RecordDataStrategy(x_path, y_path)
        elif self.mode == "learn":
            data_processing_strategy = OnlineLearningStrategy(self.max_num_data_points)
        else:
            raise ValueError(
                f"Invalid data_processing_mode: {self.mode}. Must be one of ['none', 'record', 'learn']"
            )

        # buffers to save first datapoints until min_points_before_using_gp is reached
        self._gp_buffer_x = []
        self._gp_buffer_y = []
        self._gp_warmup_done = False

        # In case we want to try an already learned model in another track
        x_train_data, y_train_data = None, None
        if self.mode == "use_prerecorded_data":
            x_train_data, y_train_data = self._load_tensor_data(x_path, y_path)

            # Apply feature selectionº
            x_train_data = self._input_feature_selector(x_train_data)

            if x_train_data.shape[0] > self.max_num_data_points:
                x_train_data = x_train_data[:self.max_num_data_points, :]
                y_train_data = y_train_data[:self.max_num_data_points, :]

        # Build GP model: full or inducing
        if not self.use_inducing_gp:
            gp_model = BatchIndependentMultitaskGPModel(
                train_x=x_train_data,
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
            gp_model = BatchIndependentInducingPointGPModel(
                train_x=x_train_data,
                train_y=y_train_data,
                likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=self._nw_gp,
                ),
                residual_dimension=self._nw_gp,
                input_dimension=sum(input_selection),
                inducing_points=gp_cfg.num_inducing_points,  # TODO: wrong initialization
                use_ard=True,
            )
            hyperparameter_names_dict = {
                "likelihood.task_noises": "likelihood.task_noises",
                "kernel.lengthscale": "covar_module.base_kernel.base_kernel.lengthscale",
                "kernel.outputscale": "covar_module.base_kernel.outputscale",
            }

        # Initialize hyperparameters if provided
        hyper_cfg = gp_cfg.hyperparameters  # TODO: load the trained hyperparams from a file
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

        # Wrap in residual model
        self.residual_gp = GPyTorchResidualModel(
            gp_model=gp_model,
            feature_selector=self._input_feature_selector,
            data_processing_strategy=data_processing_strategy,
        )

        self._prev_state_for_residual = None
        self._prev_input_for_residual = None

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
            speed_now, self.vel_y, self.measured_steer, self.yaw_rate, self.measured_acc
        ], dtype=np.float64)

        # Record new datapoints. Until we don't have at least self.min_points_before_using_gp
        if (
            self.use_gp
            and self.residual_gp is not None
            and getattr(self, "mode", None) in ("learn", "record")
        ):
            res = self._compute_residual(new_state=x0)
            if res is not None:
                x_train, residual = res

                # 1) If warm-up is not finished, collect into buffer
                if not self._gp_warmup_done:
                    self._gp_buffer_x.append(x_train)
                    self._gp_buffer_y.append(residual)

                    # When we reach the threshold, flush buffer into the GP
                    if len(self._gp_buffer_x) >= self.min_points_before_using_gp:
                        for x_b, y_b in zip(self._gp_buffer_x, self._gp_buffer_y):
                            self.residual_gp.record_datapoint(x_b, y_b)

                        # Clear buffer and mark GP as ready
                        self._gp_buffer_x.clear()
                        self._gp_buffer_y.clear()
                        self._gp_warmup_done = True

                # 2) If warm-up is finished, directly add each new datapoint
                else:
                    self.residual_gp.record_datapoint(x_train, residual)

        # Update inducing points
        self._update_inducing_points_along_horizon()

        # Delay compensation
        self.t_delay = 0.0
        propagated_x = x0

        # Fix initial state
        self.l4a_solver.set(0, "lbx", propagated_x)
        self.l4a_solver.set(0, "ubx", propagated_x)

        # Trailing control
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self.trailing_controller(self.waypoint_array_in_map[0, 2])
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
                self.l4a_solver.set(i, "lbu", np.array([-50, self.lbmpc_config.ddelta_min]))
                self.l4a_solver.set(i, "ubu", np.array([50, self.lbmpc_config.ddelta_max]))
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
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])
            print(f"[MPC Controller] Solver failed with status {status}. Applying warm start again.")

        # Actual cost of the problem, in case wanting to log
        # current_cost = self.l4a_solver.get_cost()

        # Output
        self.u0 = self.l4a_solver.get(0, "u")
        self.prev_acc = self.l4a_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.lbmpc_config.steps_delay + 1
        self.pred_x = self.l4a_solver.get(delayed_index, "x")

        self.steering_angle = self.pred_x[StateIndex.STEERING_ANGLE_DELTA.value]
        self.speed = self.pred_x[StateIndex.VELOCITY_V_X.value]
        self.acceleration = self.prev_acc + self.u0[InputIndex.JERK.value] / self.controller_frequency
        self.prev_acc = self.acceleration
        self.jerk = self.u0[InputIndex.JERK.value]

        # Update steering buffer
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.steering_angle

        self.mpc_sd = np.array([
            self.l4a_solver.get(j, "x")[:2] for j in range(self.lbmpc_config.N + 1)
        ])

        self._prev_state_for_residual = x0.copy()
        self._prev_input_for_residual = self.u0.copy()    # [ddelta, jerk]

        states_list = None
        if status == 0 or status == 2:
            states_list = []
            for node in range(self.lbmpc_config.N + 1):
                result = self.l4a_solver.get(node, "x").tolist()
                for item in result:
                    states_list.append(item)
            return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status
        else:
            return 0.0, 0.0, 0.0, self.measured_steer, states_list, status

    def apply_warm_start(self, pose_frenet: np.ndarray):
        """
        Generates and applies a simple initial guess considering constant inputs.
        """
        # TODO: add them as params
        const_acc = 1.0
        const_steer_vel = 0.0

        nx = int(self.model.n_x)
        nu = int(self.model.n_u)
        N = self.lbmpc_config.N

        x0 = np.zeros(nx)
        x0[StateIndex.POS_ON_CENTER_LINE_S.value] = pose_frenet[0]
        x0[StateIndex.MIN_DIST_TO_CENTER_LINE_N.value] = pose_frenet[1]
        x0[StateIndex.ORIENTATION_THETA.value] = pose_frenet[2]
        x0[StateIndex.VELOCITY_V_X.value] = 1.0  # TODO: perhaps max(actual_speed, 1.0) ?
        x0[StateIndex.ACCEL.value] = const_acc

        u0 = np.zeros(nu)
        u0[InputIndex.D_STEERING_ANGLE.value] = const_steer_vel

        warm_start = np.zeros((N + 1, nx))
        warm_start[0, :] = x0
        x_prev = warm_start[0, :]
        for i in range(1, N + 1):
            x_next = self._predict_next_state_nominal(x_prev, u0)
            warm_start[i, :] = x_next
            self.l4a_solver.set(i, "x", x_next)
            if i < N:
                self.l4a_solver.set(i, "u", u0)
            x_prev = x_next

    # TODO: Shouldn't we use the previous results?
    def update_warm_start(self) -> None:
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

    def propagate_time_delay(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate current state over self.t_delay with discrete steps of the nominal model.
        """
        x = states.copy()
        u = inputs.copy()

        # Number of MPC steps that correspond to the delay
        # TODO: think better way to do this: using previous inputs, using different time step if its the case
        n_steps = int(round(self.t_delay * self.controller_frequency))
        n_steps = max(n_steps, 0)

        for _ in range(n_steps):
            x = self._predict_next_state_nominal(x, u)

        # Saturations as before
        if abs(x[StateIndex.STEERING_ANGLE_DELTA.value]) > self.constraint.delta_max:
            x[StateIndex.STEERING_ANGLE_DELTA.value] = np.sign(
                x[StateIndex.STEERING_ANGLE_DELTA.value]) * self.constraint.delta_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) > self.constraint.v_x_max:
            x[StateIndex.VELOCITY_V_X.value] = np.sign(x[StateIndex.VELOCITY_V_X.value]) * self.constraint.v_x_max
        if abs(x[StateIndex.VELOCITY_V_X.value]) < self.constraint.v_x_min:
            x[StateIndex.VELOCITY_V_X.value] = self.constraint.v_x_min

        return x

    # # TODO: when acados sim validated, remove this
    # def propagate_time_delay(self, states: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    #     """
    #     RK45 forward propagation using acados model.
    #     """
    #     x0 = np.concatenate((states, [inputs[0]], [0]), axis=0)
    #     self.t_delay = 0.0

    #     solution = solve_ivp(  # TODO: use acados sim
    #         self._dynamics_of_car,
    #         t_span=[0, self.t_delay],
    #         y0=x0,
    #         method='RK45',
    #         atol=1e-4,
    #         rtol=1e-4
    #     )

    #     sol = [x[-1] for x in solution.y]

    #     # Heading saturation
    #     if abs(sol[5]) > self.constraint.delta_max:
    #         sol[5] = np.sign(sol[5]) * self.constraint.delta_max
    #     # Longitudinal velocity saturation
    #     if abs(sol[3]) > self.constraint.v_x_max:
    #         sol[3] = np.sign(sol[3]) * self.constraint.v_x_max
    #     # Min velocity
    #     if abs(sol[3]) < self.constraint.v_x_min:
    #         sol[3] = self.constraint.v_x_min

    #     return np.array(sol)[:-2]

    # TODO: when acados sim validated, remove this function
    def _dynamics_of_car(self, t, x0) -> list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''

        s = x0[0]
        n = x0[1]
        theta = x0[2]
        v_x = x0[3]
        v_y = x0[4]
        yaw_rate = x0[5]
        delta = x0[6]
        accel = x0[7]
        derDelta = x0[8]
        jerk = x0[9]
        xdot = self.model.f_expl_func(
            s, n, theta, v_x, v_y, yaw_rate, delta, accel, derDelta, jerk, self.model_params.p)

        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
                float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]), 0.0, 0.0]

        return xdot

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
        x_pred = self._predict_next_state_nominal(x_prev, u_prev)

        # Training input: [x_prev, u_prev] (full vector)
        x_train = np.concatenate((x_prev, u_prev))

        # Residual in selected coordinates
        residual = self.B_d_pinv @ (new_state - x_pred)

        return x_train, residual

    def trailing_controller(self, global_speed: float) -> float:
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

    def transform_waypoints_to_coords_from_array(self, waypoints_data: np.ndarray) -> np.ndarray:
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
        if self.sim_solver is not None:
            # Use acados sim solver created by ResidualLearningMPC
            self.sim_solver.set("x", x.astype(float))
            self.sim_solver.set("u", u.astype(float))
            self.sim_solver.set("p", np.array(self.model_params.p, dtype=float))
            status = self.sim_solver.solve()
            if status != 0:
                # fall back to Euler if sim fails
                state_with_input = np.concatenate((x, u))
                xdot = self._dynamics_of_car(0.0, state_with_input)
                Ts = 1.0 / self.lbmpc_config.MPC_freq
                return x + Ts * np.array(xdot)[: self.model.n_x]

            x_next = self.sim_solver.get("x")
            return np.array(x_next, dtype=float).reshape(-1)

        # Fallback: continuous-time model + Euler
        state_with_input = np.concatenate((x, u))
        xdot = self._dynamics_of_car(0.0, state_with_input)
        Ts = 1.0 / self.lbmpc_config.MPC_freq
        return x + Ts * np.array(xdot)[: self.model.n_x]

    def _predict_next_state_with_gp(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Learned prediction:
            x_next ≈ f_nominal(x,u) + B * μ_residual(x,u)
        Only active if use_gp and warm-up already finished.
        """
        x_next_nom = self._predict_next_state_nominal(x, u)

        if not (self.use_gp and self.residual_gp is not None and self._gp_warmup_done):
            return x_next_nom

        xu = np.concatenate((x, u))[None, :]  # shape (1, nx+nu)
        # GPyTorchResidualModel interface: TODO: check actual method name
        mu_residual, _ = self.residual_gp.predict(xu)
        mu_residual = mu_residual.detach().cpu().numpy().reshape(-1)  # (nw,)

        delta_x = self.B @ mu_residual  # (nx,)
        return x_next_nom + delta_x

    @staticmethod
    def _load_tensor_data(x_path: str, y_path: str):
        """
        Load offline data from CSV into float32 torch tensors.

        x_path: CSV with inputs, shape (N, nx+nu) or (N,)
        y_path: CSV with residuals, shape (N, nw) or (N,)
        """
        try:
            x_np = np.genfromtxt(x_path, delimiter=",")
            x_tensor = torch.from_numpy(x_np).type(torch.float32)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(-1)
        except FileNotFoundError:
            print(f"[LBMPC] File {x_path} does not exist.")
            x_tensor = None

        try:
            y_np = np.genfromtxt(y_path, delimiter=",")
            y_tensor = torch.from_numpy(y_np).type(torch.float32)
            if y_tensor.dim() == 1:
                y_tensor = y_tensor.unsqueeze(-1)
        except FileNotFoundError:
            print(f"[LBMPC] File {y_path} does not exist.")
            y_tensor = None

        return x_tensor, y_tensor


def check_file_changes(file_paths, hash_file_path) -> bool:
    """
        Return True if any of the files in `file_paths` changed since last call.

        It computes an MD5 over the concatenation of the file contents and compares
        it with the MD5 stored in `hash_file_path`. If it's different (or the file
        does not exist), it writes the new hash and returns True.

        This is the same idea as in the ZOGPMPC controller.
        """
    m = hashlib.md5()
    for fp in file_paths:
        with open(fp, "rb") as f:
            m.update(f.read())
    new_hash = m.hexdigest()

    hash_path = Path(hash_file_path)
    if hash_path.exists():
        old_hash = hash_path.read_text().strip()
        if old_hash == new_hash:
            return False

    hash_path.write_text(new_hash)
    return True
