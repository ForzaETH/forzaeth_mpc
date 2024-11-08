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
import os
import rospkg
from joblib import load
import torch
from MAP2.Sparse_GP_function import sparse_gp_model_train, sparse_gp_model_predict, sparse_gp_model_predict_fast, sparse_gp_plot
from MAP2.Sparse_GP_data_selection import data_selection, filter_by_range
from dynamic_reconfigure.client import Client
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt


class MAP2_Controller:
    def __init__(self, **kwargs):
        """
        Initialise MPC and MAP object.

        Input:  conf_file   : String containing the path to the MAP2_param_config.yaml file and MAP param
        """
        ## initialize for the upper mpc
        self.mpc_init_params(kwargs)
        # Init the mpc solver
        self.mpc_initialize_solver()

        ## initialize for the backup controller MAP
        map_init_params = kwargs['map_controller_param']
        self.trailing_config: TrailingConfig = kwargs['trailing_config']
        self.map_init_params(*map_init_params)

        # # initialize the GP model
        rospack = rospkg.RosPack()
        package_path=rospack.get_path('gp_mpc_node')
        self.model_file = os.path.join(package_path, 'src/gp_mpc_node/save_model/sparse_gp_model.pth')
        self.GP_model_load=False
        self.GP_beta_comp=np.zeros(self.N+1)

        # update the GP model through dynamic reconfigure
        self.MAP2_client = Client("MAP2_param_tuner", timeout=30)
        self.sampling_rate = 40
        self.Initial_data_size = 800
        self.Max_data_size = 1200
        self.data_duration = self.Initial_data_size / self.sampling_rate
        self.timesteps = self.data_duration * self.sampling_rate
        self.pbar = tqdm(total=self.timesteps, desc='Collecting data', ascii=True)
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.new_X = np.array([])
        self.new_Y = np.array([])
        self.inducing_points_number = 50
        self.sgp_training_iterations = 30
        self.y_min = -0.1
        self.y_max = 0.1
        self.beta_error = None #gp train y
        self.center_car_s = None # gp train x
        self.new_data_available = False
        self.counter = 0
        self.Initial_gp = False
        self.update_gp = False
        self.lap_flag = False
        self.gp_ready_preidct = False
        self.GP_predict_viz = False
        self.GP_predict_viz_Initial = False
        self.GP_predict_viz_update = False
        # Initialize the GP parameters
        warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
        self.gp_compute_timer = rospy.timer.Timer(rospy.Duration(1/40), self.gp_callback)


    def gp_callback(self, event):

        self.loop_logistic()

        # Create the initial GP model
        if self.Initial_gp == True and self.update_gp == False:
            self.Initial_GP_model_train()
            self.gp_ready_preidct = True

        # Update the GP model with new data
        if self.lap_flag == True and self.update_gp == True:
            self.gp_ready_preidct = False
            self.online_GP_model_update()
            self.gp_ready_preidct = True

    def loop_logistic(self):
        """
        Manage the loop and data collection logistics for the GP model update.
        """

        if self.lap_counter>1 and self.update_gp == False:
            self.Initial_gp = True

        # collect data to update the gp
        self.collect_lap_data()

    def collect_lap_data(self):

        # Append the current data point
        self.new_X = np.append(self.new_X, self.center_car_s)
        self.new_Y = np.append(self.new_Y, self.beta_error)

        # If a new lap is detected, store the collected data and reset
        if self.lap_flag:
            # Store the current lap data
            self.new_x = self.new_X.copy()  # Use .copy() to prevent reference issues
            self.new_y = self.new_Y.copy()

            # Reset the data for the next lap
            self.new_X = np.array([])  # Reset to an empty NumPy array
            self.new_Y = np.array([])

    def collect_Initial_GP_data(self):
        """
        Collects data during simulation.

        Adds the current state to the data array and updates the counter.
        Closes the progress bar and logs a message if data collection is complete.
        """
        if self.new_data_available:
            if self.center_car_s is not None and self.beta_error is not None:
                self.X_train = np.append(self.X_train, self.center_car_s)
                self.y_train = np.append(self.y_train, self.beta_error)
                self.counter += 1
                self.pbar.update(1)
                if self.counter == self.timesteps + 1:
                    self.pbar.close()
                    rospy.loginfo("Initial Data collection for GP is completed.")
                self.new_data_available = False

    def Initial_GP_model_train(self):
        """
        Train the initial Gaussian Process (GP) model.

        This function collects initial training data and trains the GP model once a
        sufficient number of timesteps have been reached. It then updates the model
        and visualizes the results using a non-blocking plot.

        Steps:
        1. Collects initial GP training data.
        2. Checks if the number of collected data points is equal to the specified timesteps.
        3. Reshapes the training data and fits the GP model using the provided training data.
        4. Updates the model and sets flags to indicate that the initial training is complete.
        5. Generates a plot of the GP predictions and displays it using a non-blocking pause.

        Returns:
        None

        """

        self.collect_Initial_GP_data()

        if self.counter == self.timesteps + 1:
                    #measure the time
                    start_time = rospy.get_time()
                    self.X_train = self.X_train.reshape(-1, 1)
                    self.y_train = self.y_train.reshape(-1, 1)
                    # Filter the training data by the specified range
                    self.X_train, self.y_train, _ =filter_by_range(self.X_train, self.y_train, self.y_min, self.y_max)
                    # Fit the GP regression model to the training data
                    self.model, self.likelihood = sparse_gp_model_train(self.X_train, self.y_train, self.inducing_points_number,
                                                                        self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
                    end_time = rospy.get_time()-start_time
                    print(f"GP model Initial training time: {end_time}")
                    print("The GP model is  initialized.")
                    # start to update GP
                    self.Initial_gp = False
                    self.update_gp = True
                    # First GP fit is infored to be viz
                    self.GP_predict_viz_Initial = True

    def online_GP_model_update(self):
        """
        Update the Gaussian Process (GP) model with new training data.

        This function performs the following steps:
        1. Reshapes the existing training data to the required format.
        2. Selects new data points to be added to the training dataset, ensuring the data size is within the specified limit.
        3. Fits the GP regression model to the updated training data.
        4. Optionally visualizes the updated GP model predictions in a non-blocking plot.

        Steps:
        - Data selection: Filters and updates the training data using a selection method.
        - Model fitting: Retrains the GP model with the updated training data.
        - Visualization: If enabled, plots the updated GP predictions.


        Returns:
        None
        """
        #data selection
        print("data selection is implementing. The size of new data: ", len(self.new_x))
        #measure the time
        start_time = rospy.get_time()
        self.X_train = self.X_train.reshape(-1, 1)
        self.y_train = self.y_train.reshape(-1, 1)
        self.X_train, self.y_train = data_selection(self.model, self.likelihood, self.X_train, self.y_train,
                        self.new_x, self.new_y, self.Max_data_size, self.y_min, self.y_max,
                        selection_mode="inducing_cluster")
        print("data selection is completed. The current data size: ", len(self.X_train))
        # Fit the GP regression model to the training data
        self.model, self.likelihood = sparse_gp_model_train(self.X_train, self.y_train, self.inducing_points_number,
                                                            self.sgp_training_iterations, predic_warmup=True, train_viz_disable=True)
        print("The time for updating the GP model: ", rospy.get_time() - start_time)
        # viz flag
        plt.close('all')
        self.GP_predict_viz_update = True


    def gp_model_viz(self):

        if self.GP_predict_viz_Initial == True or (self.GP_predict_viz_update == True and self.GP_predict_viz == True):
            if  self.model is not None:
                xtest = np.linspace(min(self.X_train), max(self.X_train), len(self.X_train)).reshape(-1, 1)
                gp_mean, gp_var, gp_inducing = sparse_gp_model_predict(self.model, self.likelihood, xtest)
                plt.close('all')
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sparse_gp_plot(self.X_train, self.y_train, xtest, gp_mean, gp_var, gp_inducing, ax)
                plt.tight_layout()
                plt.pause(0.01) # Pause to update the plot without blocking the code
                self.GP_predict_viz_Initial = False
                self.GP_predict_viz_update = False



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

        ### MAP2 ###
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
            # TODO : the MAP controller should be related with the warm start of the MPC
            self.logger_warn('[MAP2 Controller]: The solver did not converge. SWITCHING TO MAP')
            steering_angle, speed, acceleration = self.map_back_up_controller(
                position_in_map, waypoint_array_in_map, target_speed)

        # viz the gp model
        self.gp_model_viz()


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
                self.lap_flag = True
            else:
                self.lap_flag = False
        else:
                self.lap_flag = False

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
        # only preidc when the GP model is not updated
        if self.gp_ready_preidct == True:

            # GP conpensation for the mpc
            if self.GP_model_load and self.model is not None and self.likelihood is not None:
                    s_gp = np.array(self.mpc_sd[1:,0]%self.spline.track_length)
                    s_gp = np.append(s_gp, self.mpc_sd[self.N,0]%self.spline.track_length)
                    self.GP_beta_comp = sparse_gp_model_predict_fast(self.model, self.likelihood, s_gp)
            else:
                self.GP_beta_comp=np.zeros(self.N+1)
        else:

            self.GP_beta_comp=np.zeros(self.N+1)

    def update_online_parameters(self):
        """
        Update the online parameters for the MPC at each stafe.
        It then sets these parameters in the acados solver to be used in the optimization problem.
        """
        for i in range(self.N+1):
            idx = np.abs(self.waypoint_array_in_map[:,4]  - self.mpc_sd[i,0]%self.spline.track_length).argmin()

            # head to head racing
            # for the trailing, MPC just needs to change the reference speed (get from the trailing controller)
            if(self.state == "TRAILING"): #Trailing controller
                target_v_speed = self.trailing_controller(self.waypoint_array_in_map[0,2])
                self.overtake_d=0
            else:
                target_v_speed=np.ones(1)*self.waypoint_array_in_map[idx,2]
                self.overtake_d=0
                # head to head overtaking
                if(self.state == "OVERTAKE"):
                    self.overtake_d=self.waypoint_array_in_map[idx,3]

            online_parameters=np.array([self.weight_qn,self.weight_theta,self.weight_accel_x,self.weight_beta,self.weight_vx,self.weight_yaw_rate,target_v_speed,self.GP_beta_comp[i],0],dtype=np.float64)

            self.acados_solver.set(i,"p",online_parameters)

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
        self.states[2]=(self.center_car_s)
        self.states[3]=(self.beta_error)
        self.new_data_available = True
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


