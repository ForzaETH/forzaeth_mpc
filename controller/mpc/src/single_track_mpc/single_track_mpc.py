#! /usr/bin/env python3
from typing import Tuple
import numpy as np
import rospy
import yaml
import logging
import time
from scipy.integrate import solve_ivp
from frenet_converter.frenet_converter import FrenetConverter
from f110_msgs.msg import WpntArray
from .acados_settings import acados_settings
from ..utils.splinify import SplineTrack

class Single_track_MPC_Controller:
    def __init__(self, pose_frenet, **kwargs) -> None:
        """
        Initialise MPC object.
        
        Input:  conf_file   : String containing the path to the param_config.yaml file
        """
        # Init the parameters
        self.mpc_init_params(kwargs)

        # Init the solver
        self.mpc_initialize_solver(pose_frenet)

    def mpc_init_params(self, kwargs) -> None:
        self.racecar_version = kwargs['racecar_version']
        #MPC Params
        self.conf_params = kwargs['conf_params']
        # upper layer parameters: MPC
        self.N = self.conf_params['N']    
        t_delay = self.conf_params['t_delay']
        self.steps_delay = self.conf_params['steps_delay']
        self.t_MPC = 1 / self.conf_params['MPC_freq']
        
        rospy.loginfo(f"[MPC Controller] Steps Delay set to {self.steps_delay}. Equivalent to a delay of {1000*self.steps_delay*self.t_MPC:3.2f} milliseconds.")
        # time delay propagation
        self.t_delay = t_delay + self.t_MPC
        # print(self.conf_params)

        #seering angle buffer size
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        #Initial state
        self.mpc_sd = np.zeros((self.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s=0
        self.previous_frenet_s = 0
        #Initial time
        self.conp_time = 0

        # Init the online parameters
        mpc_init_online_params = kwargs['mpc_online_param']
        self.qadv = mpc_init_online_params[0]
        self.qv = mpc_init_online_params[1]
        self.qn = mpc_init_online_params[2]   
        self.qalpha = mpc_init_online_params[3]
        self.qac = mpc_init_online_params[4]
        self.qddelta = mpc_init_online_params[5]
        self.cs_delta_steer_min = mpc_init_online_params[6]
        self.cs_delta_steer_max = mpc_init_online_params[7]
        self.cs_delta_v_min = mpc_init_online_params[8]
        self.cs_delta_v_max = mpc_init_online_params[9]
        self.cs_v_min = mpc_init_online_params[10]
        self.cs_v_max = mpc_init_online_params[11]
        self.cs_alat_min = mpc_init_online_params[12]
        self.cs_alat_max = mpc_init_online_params[13]
    

    def mpc_initialize_solver(self, pose_frenet: np.ndarray) -> None:
        """Initialises the controller. Global waypoints are stored in a SplineTrack. All necessary parameters are stored. """
        
        rospy.loginfo(f"[MPC Controller] Waiting for global waypoints")
        shortest_path = rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.loginfo(f"[MPC Controller] Global waypoints obtained")

        x, y = self._transform_waypoints_to_cartesian(shortest_path.wpnts)
        self.fren_conv = FrenetConverter(x, y)
        
        d_left, coords_path, d_right = self._transform_waypoints_to_coords(shortest_path.wpnts) # on f track trajectory is 81.803 m long.
        
        self.spline = SplineTrack(coords_direct=coords_path)
        
        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in shortest_path.wpnts]
        vx_ref =[x.vx_mps for x in shortest_path.wpnts]
        self.s_ref = [x.s_m for x in shortest_path.wpnts]
        s0 = self.spline.params
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(s0, kapparef ,d_left, d_right, self.conf_params)
        
        # create warmstart
        self.apply_warm_start(pose_frenet=pose_frenet)
    
        self.kappa = kapparef
        
    def apply_warm_start(self, pose_frenet: np.ndarray):
        """Applies a warm start to the MPC solver.

        Args:
            pose_frenet (np.ndarray): _description_
        """
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_acc=1, const_steer_vel=0.0)
        for i in range(self.N+1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][:7])
            if i < self.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][7:])

     # main loop    
    def main_loop(self, state, position_in_map, waypoint_array_in_map, speed_now, opponent, position_in_map_frenet, single_track_state, track_length, compute_time):
        # Updating parameters from manager
        self.state = state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.vel_y = single_track_state[0]
        
        self.yaw_rate = single_track_state[1]
        self.track_length = track_length
        self.conp_time=compute_time
        # update initial state from sensors
        track_length = self.spline.track_length - 0.1 # Needed as values are not exact
        center_car_s = self.position_in_map_frenet[0] + self.conf_params["lr"]*np.cos(self.position_in_map_frenet[2])
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center =  self.position_in_map[0,2]  - np.arctan2(deriv_center[1],deriv_center[0])
        # make aplha_center between -pi and pi
        alpha_center = alpha_center % (2*np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2*np.pi
    
        if self.position_in_map_frenet[0] < self.previous_frenet_s - 1: # - 1 for rejecting noise or moving slightly backwards
            self.nr_laps += 1
        self.previous_frenet_s = self.position_in_map_frenet[0]

        current_pos_s = self.position_in_map_frenet[0]  + self.nr_laps * self.spline.track_length

        self.fre_s= current_pos_s
        self.fre_d= self.position_in_map_frenet[1]
        self.fre_alpha= alpha_center  

        x0=np.array([self.fre_s,self.fre_d,self.fre_alpha, speed_now,self.vel_y,self.steering_angle_buf[-1],self.yaw_rate],dtype=np.float64)

        # time delat compensation
        self.t_delay= self.conp_time
        self.t_delay = 0.00
        

        # print("t_delay",self.t_delay)
        # if x0[3] >= 0.1:
        #     propagated_x = self.propagate_time_delay(x0, self.u0)
        # else: 
        propagated_x = x0
        

        # set the initial state for the mpc 
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)

        # dynamically change the target speed, weight parameters and constraints
        for i in range(self.N+1):
            idx = np.abs(self.waypoint_array_in_map[:,4]  - self.mpc_sd[i,0]%self.spline.track_length).argmin()
            traget_v_speed=np.ones(1)*self.waypoint_array_in_map[idx,2]              
            if i < self.steps_delay:
                online_parameters=np.array([traget_v_speed, self.qadv, self.qv, self.qn, self.qalpha,self.qac,1e6*self.qddelta,
                                            self.cs_alat_min,self.cs_alat_max],dtype=np.float64)
            else:
                online_parameters=np.array([traget_v_speed, self.qadv, self.qv, self.qn, self.qalpha,self.qac,self.qddelta,
                                            self.cs_alat_min,self.cs_alat_max],dtype=np.float64)
                
            self.acados_solver.set(i,"p",online_parameters)

            if i < self.N:
                self.acados_solver.set(i, "lbu", np.array([self.cs_delta_v_min, self.cs_delta_steer_min]))
                self.acados_solver.set(i, "ubu", np.array([self.cs_delta_v_max, self.cs_delta_steer_max]))
                self.acados_solver.set(i+1, "lbx", np.array([self.cs_v_min, -0.4])) # do not change i+1 to i since mpc needs the initial state
                self.acados_solver.set(i+1, "ubx", np.array([self.cs_v_max, 0.4]))

         
        # Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            # if the solver fails, apply warm start again
            # the initial warm start is a bit rough, but it does the job
            self.apply_warm_start(pose_frenet=[self.fre_s,self.fre_d,self.fre_alpha])
            

        # get solution
        self.u0 = self.acados_solver.get(0, "u")
        delayed_index = self.steps_delay + 1
        self.pred_x = self.acados_solver.get(delayed_index, "x")
        self.steering_angle = self.pred_x[5] # propagated_x[5] + self.u0[1]/40 #
        self.speed = self.pred_x[3] # propagated_x[3] + self.u0[0]/40 #
        self.acceleration= self.u0[0]
        self.jerk=0
        # steering buffer
        # shift the elements in the buffer to the right and add the predicted steering angle
        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[5]
        self.states = None

        ##### visualization #####
        # Creating waypoint array with predicted positions
        self.mpc_sd = np.array([
            self.acados_solver.get(j, "x")[:2] for j in range(self.N+1)
        ])
        if status == 0:
            states = []
            for node in range(self.N+1):
                result = self.acados_solver.get(node,"x").tolist()
                for item in result:
                    states.append(item)
            self.states = states      
        return self.speed, self.acceleration, self.jerk, self.steering_angle, self.states
    #############
    # Utilities #
    #############
    def get_warm_start(self, pose_frenet: np.ndarray, const_acc: float, const_steer_vel: float)->np.array:
        """
        Returns a warm start trajectory for the MPC. This is done by propagating the current state with a constant acceleration and steering angle.
        
        Input:  const_acc   : Constant acceleration
                const_steer : Constant steering angle
        
        Returns: np.array    : Warm start trajectory
        """
        warm_start = np.zeros((self.N+1, 9))
        warm_start[0] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2], 1, 0, 0, 0, const_acc, const_steer_vel])
        for i in range(1, self.N+1):
            der_state = self._dynamics_of_car(0, warm_start[i-1])
            warm_start[i] = warm_start[0] + np.array(der_state)/20 # ugly euler integration
        return warm_start
    
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
    
    def propagate_time_delay(self, states: np.array, inputs:np.array)->np.array:
        """
        RK45 forward propagation. Calls the function _dynamics_of_car which contains the dynamics of the car.

        Input:  np.array    : Holds the current state which needs to be propagated.
                inputs      : Holds the current inputs from the MPC.

        Returns: np.array   : Propagated state without input. 
        
        """


        # Initial condition on the ODE
        x0 = np.concatenate((states,[inputs[0]],[0]), axis=0)
        # forward propagation
        self.t_delay=0.0
        # DEBUG print nicely formatted aligned state
        # print("BEFORE state:", end=" ")
        # for x in x0:
        #     print(f"{x:5.2f}", end=" ")
        # print("\n")
        
        solution = solve_ivp(self._dynamics_of_car, t_span= [0, self.t_delay], y0=x0, method='RK45', atol=1e-4, rtol=1e-4)
        # NOTE: can't really converge? seen with negative acceleration

        # DEBUG print nicely formatted aligned state
        # print("AFTER  state:", end=" ")
        # for x in solution.y[:,-1]:
        #     print(f"{x:5.2f}", end=" ")
        # print("\n")
        
        solution = [x[-1] for x in solution.y]
        # Constraint on max. steering angle
        if abs(solution[5]) > self.model.delta_max:
            solution[5] = np.sign(solution[5]) * self.model.delta_max
        
        # Constraint on max. vel
        if abs(solution[3]) > self.constraint.v_x_max:
            solution[3] = np.sign(solution[3]) * self.constraint.v_x_max
        
        # On minimum speed
        if abs(solution[3]) < self.constraint.v_x_min:
            solution[3] = self.constraint.v_x_min
        
      
        # Only get the state as solution of where the car will be in t_delay seconds
        # not including the input
        # states[5] = solution[5]
        # states[6] = solution[6]
        # return states
        return np.array(solution)[:-2]
    
    def _dynamics_of_car(self, t, x0)->list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''
    
        s = x0[0]
        n = x0[1]
        theta = x0[2]
        v_x = x0[3]
        v_y = x0[4]
        delta = x0[5]
        yaw_rate = x0[6]
        accel = x0[7]
        derDelta = x0[8]
        xdot = self.model.f_expl_func(s, n, theta, v_x, v_y, delta, yaw_rate, accel, derDelta, self.model_params.p)
        
        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]), float(xdot[4]),float(xdot[5]),float(xdot[6]),accel,derDelta]  

        return xdot
