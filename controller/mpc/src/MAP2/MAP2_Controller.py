#!/usr/bin/env python3
from typing import Tuple
import logging
import numpy as np
import rospy
from steering_lookup.lookup_steer_angle import LookupSteerAngle
from . fre_point_mass_model_acados_settings  import acados_settings
from pbl_config import  TrailingConfig
from .utils.splinify import SplineTrack
from f110_msgs.msg import WpntArray

# for the GP_for_MPC_node
from MAP2.GP_Model import GPModel  # Import GPModel class
from MAP2.Sparse_GP_function import  sparse_gp_model_predict_fast



class MAP2_Controller:
    def __init__(self, **kwargs):
        """
        Initialise MPC and MAP object.

        Input:  conf_file   : String containing the path to the MAP2_param_config.yaml file and MAP param
        """

        # Initialize MPC and MAP parameters
        self.initialize_mpc(kwargs)
        self.initialize_map(kwargs)

        # Initialize the GP prediction values
        self.GP_beta_comp=np.zeros(self.N+1)
        self.beta_error = None #gp train y
        self.center_car_s = None # gp train x
        self.gp_model = GPModel()  # Instantiate GPModel

        # Initialize timers and other configurations
        self.gp_compute_timer = rospy.timer.Timer(rospy.Duration(1/40), self.gp_model.gp_callback)

    def initialize_mpc(self, kwargs):
        """
        Initialize MPC parameters and solver.

        Args:
            kwargs: Dictionary containing parameters for the MPC initialization.
        """
        self.mpc_init_params(kwargs)
        self.mpc_initialize_solver()

    def initialize_map(self, kwargs):
        """
        Initialize MAP controller parameters.

        Args:
            kwargs: Dictionary containing parameters for the MAP initialization.
        """
        map_init_params = kwargs['map_controller_param']
        self.trailing_config: TrailingConfig = kwargs['trailing_config']
        self.map_init_params(*map_init_params)

    def mpc_init_params(self, kwargs) -> None:
        self.racecar_version = kwargs['racecar_version']
        #MPC Params
        self.conf_params = kwargs['conf_params']

        # upper layer parameters: MPC
        self.N = self.conf_params['N']

        # Initialize the weight parameters for the MPC
        self.mpc_weight_params(self.conf_params)

        # initial s
        self.fre_s = 0
        self.mpc_sd = np.zeros((self.N + 1, 4))
        self.pre_s =0
        self.lap_counter=0
        self.states = np.zeros((self.N + 1, 4))

        # head to head parameter
        self.overtake_d=0

        #initial trailing parameter
        self.loop_rate = 40
        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0

    def mpc_weight_params(self, conf_params) -> None:
        # Parameters from manager
        self.time_delay_step = conf_params['time_delay_step']
        self.t_d_c = conf_params['t_d_c']
        self.online_ac_y_max= conf_params['online_ac_y_max']
        self.online_ac_x_max= conf_params['online_ac_x_max']
        self.weight_qn= conf_params['weight_qn']
        self.weight_theta= conf_params['weight_theta']
        self.weight_accel_x= conf_params['weight_accel_x']
        self.weight_beta= conf_params['weight_beta']
        self.weight_vx= conf_params['weight_vx']
        self.weight_yaw_rate= conf_params['weight_yaw_rate']



    def mpc_initialize_solver(self) -> None:
        """Initialises the controller. Global waypoints are stored in a SplineTrack. All necessary parameters are stored. """

        rospy.loginfo(f"[MPC Controller] Waiting for global waypoints")
        mincurv_raceline = rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.loginfo(f"[MPC Controller] Global waypoints obtained")

        x, y = self._transform_waypoints_to_cartesian(mincurv_raceline.wpnts)
        d_left, coords_path, d_right = self._transform_waypoints_to_coords(mincurv_raceline.wpnts) # on f track trajectory is 81.803 m long.

        self.spline = SplineTrack(coords_direct=coords_path)

        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in mincurv_raceline.wpnts]
        vx_ref = [x.vx_mps for x in mincurv_raceline.wpnts]
        self.s_ref = np.array([x.s_m for x in mincurv_raceline.wpnts])
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(self.s_ref, kapparef,vx_ref ,d_left, d_right, self.conf_params)
        self.kappa = kapparef
        self.lat_acc=0

    def map_init_params(self,
                t_clip_min,
                t_clip_max,
                m_map,
                q_map,
                racecar_version,
                floor,
                )-> None:
        # Parameters from manager
        self.t_clip_min = t_clip_min
        self.t_clip_max = t_clip_max
        self.m_map = m_map
        self.q_map = q_map
        self.racecar_version = racecar_version
        self.floor = floor

        # loggers
        self.logger_info = rospy.loginfo
        self.logger_warn = rospy.logwarn
        self.map_lat_acc=0
        self.steer_lookup = LookupSteerAngle(self.racecar_version, self.floor, "pacejka",self.logger_info)


    # main loop
    def main_loop(self, machine_state, position_in_map, waypoint_array_in_map,state,position_in_map_frenet,comp_time,opponent):
        position_in_map = position_in_map.squeeze() #squeeze dim

        # Desired speed at waypoint closest to car
        self.state=machine_state
        self.waypoint_array_in_map=waypoint_array_in_map
        target_speed = waypoint_array_in_map[0][2] # local waypoint: x, y, speed, d, s, kappa, psi, ax
        self.comp_time = comp_time
        self.vx_now = state[0]
        self.vy_now = state[1]
        self.yaw_rate_now = state[2]
        self.position_in_map = position_in_map
        self.position_in_map_frenet = position_in_map_frenet
        self.v=np.sqrt(self.vx_now**2+self.vy_now**2)
        self.opponent = opponent


        ### MAP2 ### #TODO: there is a lot of space to reduce the computation time. I guess around 5~6ms plus gp around 8ms
        # set the initial state for the mpc
        propagated_x = self.get_initial_mpc_state()
        # propagated_x=np.array([self.fre_s, self.fre_d, self.fre_alpha, self.v],dtype=np.float64)
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        #measure the time
        # GP model loads and compensates the error
        self.sparse_gp_model_predict_for_mpc()

        # update the online parameters
        self.update_online_parameters()

        # Solve optimal upper layer control problem
        status = self.acados_solver.solve()

        if status == 0:
            # run MAP2 controller
            steering_angle, speed, acceleration = self.obtain_MAP2_control_command()
        else:
            # run MAP controller
            # TODO : the MAP controller should be replaced with the warm start of the MPC
            self.logger_warn('[MAP2 Controller]: The solver did not converge. SWITCHING TO MAP')
            steering_angle, speed, acceleration = self.map_back_up_controller(
                position_in_map, waypoint_array_in_map, target_speed)

        # viz the gp model
        self.gp_model.gp_model_viz()
        return speed, acceleration, steering_angle,  self.states

    @staticmethod
    def distance(point1, point2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            point1 (tuple): A tuple containing the x and y coordinates of the first point.
            point2 (tuple): A tuple containing the x and y coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return (((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))**0.5

    @staticmethod
    def nearest_waypoint(position, waypoints):
        """
        Finds the index of the nearest waypoint to a given position.

        Args:
            position (tuple): A tuple containing the x and y coordinates of the position.
            waypoints (numpy.ndarray): An array of tuples representing the x and y coordinates of waypoints.

        Returns:
            int: The index of the nearest waypoint to the position.
        """
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position)

    def waypoint_at_distance_infront_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Finds the waypoint a given distance in front of a given waypoint.

        Args:
            distance (float): The distance to travel from the given waypoint.
            waypoints (numpy.ndarray): An array of tuples representing the x and y coordinates of waypoints.
            idx_waypoint_behind_car (int): The index of the waypoint behind the car.

        Returns:
            numpy.ndarray: A tuple containing the x and y coordinates of the waypoint a given distance in front of the given waypoint.
        """
        dist = 0
        i = idx_waypoint_behind_car

        while dist < distance:
            i = (i + 1) % len(waypoints)
            dist = self.distance(waypoints[idx_waypoint_behind_car], waypoints[i])

        return np.array(waypoints[i])

    def _transform_waypoints_to_cartesian(self, data: WpntArray)-> Tuple[np.ndarray, np.ndarray]:
        """Helper function to store the received waypoints into the right format such that they can be used for initialisation of SplineTrack.

        Input: data: WpntArray: Holds the data received from the globalwaypoints.

        Returns: Tuple: Stores the waypoints as follows: Left boundaries, reference path, right boundaries
        """
        x = np.zeros(len(data))
        y = np.zeros(len(data))
        for idx, wpnt in enumerate(data):
            x[idx] = wpnt.x_m
            y[idx] = wpnt.y_m
        return np.array(x), np.array(y)

    def _transform_waypoints_to_coords(self, data: WpntArray)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper function to store the received waypoints into the right format such that they can be used for initialisation of SplineTrack.

        Input: data: WpntArray: Holds the data received from the globalwaypoints.

        Returns: Tuple: Stores the waypoints as follows: Left boundaries, reference path, right boundaries
        """
        waypoints = np.zeros((len(data), 2))
        d_left = np.zeros(len(data))
        d_right = np.zeros(len(data))
        boundaries = np.zeros((len(data), 2))
        for idx, wpnt in enumerate(data):
            waypoints[idx] = [wpnt.x_m, wpnt.y_m]
            d_left[idx] = wpnt.d_right              # Fix for boundaries due to frenet cartesian conversion
            d_right[idx] = wpnt.d_left              # Fix for boundaries due to frenet cartesian conversion
        res_coords = np.array([boundaries[:-1],waypoints[:-1],boundaries[:-1]])
        return d_left, res_coords, d_right

    def get_initial_mpc_state(self):
        """
        Calculate and return the initial state for the MPC based on the current vehicle position.

        Returns:
            propagated_x (np.ndarray): The initial state for the MPC including s, d, alpha, and velocity.
        """
        self.yaw=self.position_in_map[2]
        center_car_s = self.position_in_map_frenet[0] +self.conf_params["lr"]*np.cos(self.position_in_map_frenet[2])
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center =  self.yaw  - np.arctan2(deriv_center[1],deriv_center[0])

        # make aplha_center between -pi and pi
        alpha_center = alpha_center % (2*np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2*np.pi

        # continuous s
        if center_car_s-self.pre_s < -0.5*self.spline.track_length:
            if abs(center_car_s) < 1:
                self.lap_counter += 1
                # Update the lap counter in the GPModel
                self.gp_model.lap_counter = self.lap_counter
                self.gp_model.lap_flag = True
            else:
                self.gp_model.lap_flag = False
        else:
                self.gp_model.lap_flag = False

        self.pre_s = center_car_s
        continue_s = center_car_s + self.lap_counter * self.spline.track_length

        # update the initial state for the mpc
        self.fre_s= continue_s
        self.fre_d= self.position_in_map_frenet[1]
        self.fre_alpha= alpha_center
        self.center_car_s = center_car_s

        propagated_x=np.array([self.fre_s, self.fre_d, self.fre_alpha, self.v],dtype=np.float64)

        return propagated_x

    def sparse_gp_model_predict_for_mpc(self):
        """
        Performs sparse Gaussian Process (GP) model prediction to provide compensation for the MPC controller.

        Function workflow:
        1. First checks the `gp_ready_predict` flag to ensure that prediction is only executed when the GP model is updated and ready.
        2. If the GP model is loaded and both model and likelihood objects are available, it calculates positions on the MPC trajectory and performs GP compensation prediction.
        Otherwise, it returns a zero array as compensation, indicating no GP compensation.

        Returns:
        - Updates `self.GP_beta_comp`: If the GP model is available, it contains compensation values; otherwise, it is set to a zero array.
        """
        # Only predict when the GP model is updated and ready
        if self.gp_model.gp_ready_preidct:

            # GP compensation for the MPC
            if self.gp_model.GP_model_load and self.gp_model.model is not None and self.gp_model.likelihood is not None:
                s_gp = np.array(self.mpc_sd[1:, 0] % self.spline.track_length)
                s_gp = np.append(s_gp, self.mpc_sd[self.N, 0] % self.spline.track_length)
                self.GP_beta_comp = sparse_gp_model_predict_fast(self.gp_model.model, self.gp_model.likelihood, s_gp)
            else:
                self.GP_beta_comp = np.zeros(self.N + 1)
        else:
            self.GP_beta_comp = np.zeros(self.N + 1)


    def update_online_parameters(self):
        """
        Update the online parameters for the MPC at each stafe.
        It then sets these parameters in the acados solver to be used in the optimization problem.
        """
        # Precompute the target point indices for each mpc_sd value, reducing calculations inside the loop
        target_points = self.mpc_sd[:, 0] % self.spline.track_length
        indices = np.abs(self.waypoint_array_in_map[:, 4][:, None] - target_points).argmin(axis=0)

        # Determine target speed and overtaking distance based on the current state
        if self.state == "TRAILING":
            # In trailing mode, set speed based on the trailing controller
            target_v_speed = self.trailing_controller(self.waypoint_array_in_map[0, 2])
            self.overtake_d = 0
        elif self.state == "OVERTAKE":
            # In overtaking mode, get speed and distance values from waypoint data
            target_v_speed = self.waypoint_array_in_map[indices, 2]
            self.overtake_d = self.waypoint_array_in_map[indices, 3]
        else:
            # For normal operation, get speed from waypoint data and set overtaking distance to zero
            target_v_speed = self.waypoint_array_in_map[indices, 2]
            self.overtake_d = 0

        # Prepare the constant weights array
        weights = np.array([self.weight_qn, self.weight_theta, self.weight_accel_x, self.weight_beta,
                            self.weight_vx, self.weight_yaw_rate], dtype=np.float64)

        # Set parameters for each timestep in a vectorized way
        for i in range(self.N + 1):
            # Concatenate weights, target speed, and GP compensation terms for each step
            online_parameters = np.hstack((weights, target_v_speed[i], self.GP_beta_comp[i], self.overtake_d))
            self.acados_solver.set(i, "p", online_parameters)

    def obtain_MAP2_control_command(self):
        """
        Calculate and return the control command for the vehicle using the MPC solution.

        The lateral acceleration is used to calculate the steering angle using the lookup table.

        Returns:
            steering_angle (float): The calculated steering angle.
            speed (float): The target speed for the vehicle.
            acceleration (float): The calculated acceleration.
        """
        # Get the optimal state, control input, and lateral acceleration
        # TODO there seems to be a confilct between the acados solver and the first gpytorch training
        self.mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:4] for j in range(self.N+1)
        ])
        u_sd = np.array([
            self.acados_solver.get(j, "u")[:2] for j in range(self.N)
        ])
        acc_sd=np.array([
            self.mpc_sd[j+1,3]*u_sd[j,1] for j in range(self.N)
        ])

        time_delay_step=round(self.time_delay_step)
        self.lat_acc = acc_sd[time_delay_step]

        map2_target_speed = self.mpc_sd[time_delay_step+1,3]
        acceleration = u_sd[time_delay_step,0]

        steering_angle = self.steer_lookup.lookup_steer_angle(self.lat_acc, map2_target_speed)
        speed = np.max(map2_target_speed, 0)  # no negative speed

        # Publish mpc states for viz and publish gp model data
        self.states = np.concatenate(
            [ self.acados_solver.get(node, "x") for node in range(self.N + 1)
            ]).tolist()

        # TODO : the gp training data should be published in a single topic
        # Prevent division by zero or very small numbers
        if abs(self.vx_now) > 0.1:  # Use a small threshold to avoid numerical instability
            beta_real = np.arctan(self.vy_now / self.vx_now)
        else:
            beta_real = 0  # When speed is too small, assume no slip angle

        beta=np.arcsin(self.conf_params['lr']*u_sd[0,1]/(self.mpc_sd[0,3]+0.5))
        self.beta_error = beta_real-beta
        # update the gp model
        self.gp_model.center_car_s = self.center_car_s
        self.gp_model.beta_error = self.beta_error
        self.gp_model.new_data_available = True
        # data saving for rosbag
        self.states[2]=(self.center_car_s)
        self.states[3]=(self.beta_error)
        self.states[6]=(self.gp_model.gp_train_time)
        self.states[7]=(self.gp_model.gp_selction_time)

        return steering_angle, speed, acceleration


    def map_back_up_controller(self, position_in_map, waypoint_array_in_map, target_speed):

        #MAP controller:
        # Calculate lookahead distance
        # Define lookahead distance as an affine function with  tuning parameter m and q
        idx_nearest_waypoint = self.nearest_waypoint(position_in_map[:2], waypoint_array_in_map[:, :2])
        lookahead_distance = self.q_map + target_speed * self.m_map
        lookahead_distance = np.clip(lookahead_distance, self.t_clip_min, self.t_clip_max)
        lookahead_point = self.waypoint_at_distance_infront_car(lookahead_distance, waypoint_array_in_map[:,:2], idx_nearest_waypoint)


        # map lateral acceleration
        # Vector from the current position to the point at lookahead distance
        position_la_vector = np.array([lookahead_point[0] - position_in_map[0], lookahead_point[1] - position_in_map[1]])
        yaw = position_in_map[2]
        eta = np.arcsin(np.dot([-np.sin(yaw), np.cos(yaw)], position_la_vector)/np.linalg.norm(position_la_vector))
        self.map_lat_acc = 2*target_speed**2 / lookahead_distance * np.sin(eta)

        steering_angle = self.steer_lookup.lookup_steer_angle(self.map_lat_acc, target_speed)
        speed = np.max(target_speed, 0)  # no negative speed
        acceleration = 0

        return steering_angle, speed, acceleration


    def trailing_controller(self, global_speed):

        self.gap = (self.opponent[0] - self.position_in_map_frenet[0])%self.spline.track_length # gap to opponent
        # prev_gap = self.gap if self.gap_actual is None else self.gap_actual
        self.gap_actual = self.gap
        if self.trailing_config.trailing_mode:
            self.gap_should = self.trailing_config.trailing_gap
        else:
            self.gap_should = self.position_in_map_frenet[2] * self.trailing_config.trailing_gap

        self.gap_error = self.gap_should - self.gap_actual
        self.v_diff =  self.position_in_map_frenet[2] - self.opponent[2]
        self.i_gap = np.clip(self.i_gap + self.gap_error/self.loop_rate, -10, 10)

        p_value = self.gap_error * self.trailing_config.trailing_p_gain
        d_value = self.v_diff * self.trailing_config.trailing_d_gain
        i_value = self.i_gap * self.trailing_config.trailing_i_gain


        self.trailing_command = np.clip(self.opponent[2] - p_value - i_value - d_value, 0, global_speed)
        if not self.opponent[4] and self.gap_actual > self.gap_should:
            self.trailing_command = max(self.trailing_config.blind_trailing_speed, self.trailing_command)

        return self.trailing_command


