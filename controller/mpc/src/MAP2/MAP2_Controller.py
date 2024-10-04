#!/usr/bin/env python3
from typing import Tuple
import logging
import numpy as np
import rospy
from steering_lookup.lookup_steer_angle import LookupSteerAngle
from . fre_point_mass_model_acados_settings  import acados_settings
from .utils.splinify import SplineTrack
from f110_msgs.msg import WpntArray

# for the GP_for_MPC_node
import os
import rospkg
from joblib import load
from gp_mpc_node.gp import CustomGPRegression, CustomKernelFunctions

class MAP2_Controller:
    def __init__(self, **kwargs):
        """
        Initialise MPC and MAP object.

        Input:  conf_file   : String containing the path to the MAP2_param_config.yaml file and MAP param
        """
        ## initialize for the upper mpc
        # Init the mpc parameters
        self.mpc_init_params(kwargs)

        # Init the mpc solver
        self.mpc_initialize_solver()

        ## initialize for the lower controller:MAP
        map_init_params = kwargs['map_controller_param']
        map2_init_params = kwargs['map2_online_param']
        self.map2_init_params(*map2_init_params)
        self.map_init_params(*map_init_params)

        # # load the GP model
        rospack = rospkg.RosPack()
        package_path=rospack.get_path('gp_mpc_node')
        self.model_file = os.path.join(package_path, 'src/gp_mpc_node/save_model/gp_model.pkl')
        self.gp_model=load(self.model_file)
        self.GP_model_load=self.conf_params["GP_model_load"]
        self.GP_model_update=self.conf_params["GP_model_update"]
        self.GP_model_update_flag=self.GP_model_update
        self.GP_beta_comp=0

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

    def map2_init_params(self, t_d_v,
                         t_d_c,
                         online_ac_y_max,
                         online_ac_x_max,
                         weight_qn,
                         weight_theta,
                         weight_accel_x,
                         weight_beta,
                         weight_vx,
                         weight_yaw_rate,
                         ) -> None:
        # Parameters from manager
        self.t_d_v=t_d_v
        self.t_d_c=t_d_c
        self.online_ac_y_max=online_ac_y_max
        self.online_ac_x_max=online_ac_x_max
        self.weight_qn=weight_qn
        self.weight_theta=weight_theta
        self.weight_accel_x=weight_accel_x
        self.weight_beta=weight_beta
        self.weight_vx=weight_vx
        self.weight_yaw_rate=weight_yaw_rate
        self.lat_acc=0

    def mpc_init_params(self, kwargs) -> None:
        self.racecar_version = kwargs['racecar_version']
        #MPC Params
        self.conf_params = kwargs['conf_params']

        # upper layer parameters: MPC
        self.N = self.conf_params['N']
        t_delay = self.conf_params['t_delay']
        self.t_MPC = 1 / self.conf_params['MPC_freq']
        # time delay propagation
        self.t_delay = t_delay
        print(self.conf_params)

        #Initial time
        self.conp_time = 0

        # initial s
        self.fre_s = 0
        self.mpc_sd = np.zeros((self.N + 1, 4))
        self.pre_s =0
        self.lap_counter=0



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


    # main loop
    def main_loop(self, position_in_map, waypoint_array_in_map,state,position_in_map_frenet,comp_time):
        position_in_map = position_in_map.squeeze() #squeeze dim
        # Desired speed at waypoint closest to car
        self.waypoint_array_in_map=waypoint_array_in_map
        target_speed = waypoint_array_in_map[0][2] # local waypoint: x, y, speed, d, s, kappa, psi, ax
        self.comp_time = comp_time
        self.vx_now = state[0]
        self.vy_now = state[1]
        self.yaw_rate_now = state[2]
        self.position_in_map = position_in_map
        self.position_in_map_frenet = position_in_map_frenet
        self.v=np.sqrt(self.vx_now**2+self.vy_now**2)

        ### MAP2 ###
        #Upper curvature mpc:
        # update initial state from sensors

        # calculate the current alpha angle
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
        self.pre_s = center_car_s
        continue_s = center_car_s + self.lap_counter * self.spline.track_length

        # update the initial state for the mpc
        self.fre_s= continue_s
        self.fre_d= self.position_in_map_frenet[1]
        self.fre_alpha= alpha_center

        # set the initial state for the mpc
        propagated_x=np.array([self.fre_s, self.fre_d, self.fre_alpha, self.v],dtype=np.float64)
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        # GP conpensation for the mpc
        if self.GP_model_load:
            if self.GP_model_update and not self.GP_model_update_flag:
                self.gp_model=load(self.model_file)
                self.GP_model_update_flag=True
                print("update the GP model")
            if not self.GP_model_update:
                self.GP_model_update_flag = False

        for i in range(self.N+1):
            idx = np.abs(self.waypoint_array_in_map[:,4]  - self.mpc_sd[i,0]%self.spline.track_length).argmin()
            traget_v_speed=np.ones(1)*self.waypoint_array_in_map[idx,2]
            if self.GP_model_load:
                if i<self.N:
                    self.GP_beta_comp=self.gp_model.predict(np.array(self.mpc_sd[i+1,0]%self.spline.track_length))
                else:
                    self.GP_beta_comp=self.gp_model.predict(np.array(self.mpc_sd[i,0]%self.spline.track_length))
            online_parameters=np.array([self.weight_qn,self.weight_theta,self.weight_accel_x,self.weight_beta,self.weight_vx,self.weight_yaw_rate,traget_v_speed,self.GP_beta_comp],dtype=np.float64)

            self.acados_solver.set(i,"p",online_parameters)
        # Solve optimal upper layer control problem
        status = self.acados_solver.solve()

        # Get the optimal state, control input, and lateral acceleration
        mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:4] for j in range(self.N+1)
        ])
        self.mpc_sd=mpc_sd
        u_sd = np.array([
            self.acados_solver.get(j, "u")[:2] for j in range(self.N)
        ])
        acc_sd=np.array([
            mpc_sd[j+1,3]*u_sd[j,1] for j in range(self.N)
        ])

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

        time_delay_step=round(self.t_d_v)
        self.lat_acc_0 = acc_sd[time_delay_step]

        self.lat_acc = self.lat_acc_0
        map2_target_speed = mpc_sd[1,3]
        acceleration = u_sd[0,0]

        # If the solver did not converge, switch to MAP
        if status != 0:
            self.logger_warn('[MAP2 Controller]: The solver did not converge. SWITCHING TO MAP')
            self.lat_acc = self.map_lat_acc
            map2_target_speed=target_speed
            acceleration=0

        steering_angle = self.steer_lookup.lookup_steer_angle(self.lat_acc, map2_target_speed)
        speed = np.max(map2_target_speed, 0)  # no negative speed

        # Publish mpc states for viz
        if status == 0:
            states = []
            for node in range(self.N+1):
                result = self.acados_solver.get(node,"x").tolist()
                for item in result:
                    states.append(item)
            self.states = states

        # Prevent division by zero or very small numbers
        if abs(self.vx_now) > 1e-6:  # Use a small threshold to avoid numerical instability
            beta_real = np.arctan(self.vy_now / self.vx_now)
        else:
            beta_real = 0  # When speed is too small, assume no slip angle
        beta=np.arcsin(self.conf_params['lr']*u_sd[0,1]/(mpc_sd[0,3]+0.5))
        self.states[2]=(center_car_s)
        self.states[3]=(beta_real-beta)
        return speed, acceleration, steering_angle, lookahead_point, lookahead_distance, idx_nearest_waypoint, self.states

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

