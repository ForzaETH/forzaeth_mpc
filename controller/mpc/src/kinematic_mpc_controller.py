#! /usr/bin/env python3

import os
import time
from typing import Tuple

import numpy as np
import rospkg
import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from dynamic_reconfigure.msg import Config
from f110_msgs.msg import WpntArray
from f110_msgs.msg import kinematic_mpc_param      
from frenet_conversion.srv import Frenet2Glob, Glob2Frenet
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from kinematic_mpc.acados_settings import acados_settings
from plotting_fnc import *
from scipy.integrate import solve_ivp
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray
from tf.transformations import euler_from_quaternion
from utils.indecies_old import Parameter, State
from utils.splinify import SplineTrack
from visualization_msgs.msg import Marker, MarkerArray
from frenet_converter.frenet_converter import FrenetConverter

class MPCNode:
    def __init__(self,) -> None:
        """
        Initialise MPC object.
        
        Input:  conf_file   : String containing the path to the param_config.yaml file
        """

        rospy.init_node("mpc_node", anonymous=True, log_level=rospy.DEBUG)

        self._init_params()

        self.vel_x = 0
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)
        
        self.glob2frenet = rospy.ServiceProxy('convert_glob2frenet_service', Glob2Frenet)
        self.frenet2glob = rospy.ServiceProxy('convert_frenet2glob_service', Frenet2Glob)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._odom_frenet_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, self._input_cb)
        rospy.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self._global_velocity_scale_cb)
        rospy.Subscriber("/mpc_param_tuner/parameter_updates", Config, self.mpc_params_cb) #l1 param tuning/updating
        
        self.mpc_param_pub = rospy.Publisher("/mpc_controller/param", kinematic_mpc_param, queue_size=10)
        self.pred_pos_pub = rospy.Publisher("/mpc_controller/predicted_position", MarkerArray, queue_size=10)
        self.target_pos_pub = rospy.Publisher("/mpc_controller/target_position", MarkerArray, queue_size=10)
        self.drive_pub = rospy.Publisher('/mpc_controller/input_stream', AckermannDriveStamped, queue_size=10)
        self.drive_next_pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        self.d_left_pub = rospy.Publisher("/mpc_controller/d_left", MarkerArray, queue_size=10)
        self.d_mid_pub = rospy.Publisher("/mpc_controller/d_mid", MarkerArray, queue_size=10)
        self.d_right_pub = rospy.Publisher("/mpc_controller/d_right", MarkerArray, queue_size=10)
        self.pos_pub = rospy.Publisher("/mpc_controller/current_position", Marker, queue_size=5)
        self.pos_n_pub = rospy.Publisher("/mpc_controller/current_pos_n", Odometry, queue_size=10)
        self.comp_time_pub = rospy.Publisher("/mpc_controller/comp_time", Float64, queue_size=10 )
        self.states_pub = rospy.Publisher("/mpc_controller/states", Float64MultiArray, queue_size=10)
        self.states_propagated_pub = rospy.Publisher("/mpc_controller/states_propagated", Float64MultiArray, queue_size=10)


        self._initialize_solver()

    def _init_params(self,) -> None:

        # load the yaml of mpc
        dir_path = rospkg.RosPack().get_path("stack_master")
        self.racecar_version = rospy.get_param("/racecar_version")
        file_path = rospy.get_param("/mpc_config_filename")
        self.conf_file = os.path.join(dir_path, "config", self.racecar_version, file_path)
        with open(self.conf_file, "r") as conf_file:
            self.conf_params = yaml.safe_load(conf_file)
        
        # N:prediction horizon
        self.N = self.conf_params['N']
        
        # Just for cost function testing
        self.track_safety_margin = self.conf_params['track_safety_margin']
        t_delay = self.conf_params['t_delay']
        self.t_MPC = 1 / self.conf_params['MPC_freq']
        self.conp_time = 0

        # time delay propagation
        self.t_delay = t_delay + self.t_MPC

        print(self.conf_params)

    def _initialize_solver(self) -> None:
        """Initialises the controller. Global waypoints are stored in a SplineTrack. All necessary parameters are stored. """
        
        rospy.loginfo(f"[MPC Controller] Waiting for global waypoints")
        shortest_path = rospy.wait_for_message("/global_waypoints", WpntArray)
        rospy.loginfo(f"[MPC Controller] Global waypoints obtained")

        x, y = self._transform_waypoints_to_cartesian(shortest_path.wpnts)
        self.fren_conv = FrenetConverter(x, y)
        d_left, coords_path, d_right = self._transform_waypoints_to_coords(shortest_path.wpnts) # on f track trajectory is 81.803 m long.
        
        # create a spline class
        self.spline = SplineTrack(coords_direct=coords_path)
        
        self.nr_laps = 0
      
        kapparef = [x.kappa_radpm for x in shortest_path.wpnts]
        vx_ref =[x.vx_mps for x in shortest_path.wpnts]
        self.s_ref = [x.s_m for x in shortest_path.wpnts]
        # spline.params: an array of each length in the track
        s0 = self.spline.params
        
        #substitute parameters into the acados
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(s0, kapparef,vx_ref, d_left, d_right, self.conf_params)

        self.kappa = self.model.kappa

        # Set the initial value for online parameter tuning
        self.qadv = self.conf_params["qadv"]
        self.qv = self.conf_params["qv"]
        self.qn = self.conf_params["qn"]   
        self.qalpha = self.conf_params["qalpha"]
        self.cs_v_min = self.conf_params["v_min"]
        self.cs_v_max = self.conf_params["v_max"]
        self.cs_delta_steer_min = self.conf_params["ddelta_min"]
        self.cs_delta_steer_max = self.conf_params["ddelta_max"]
        self.cs_delta_v_min = self.conf_params["a_min"]
        self.cs_delta_v_max = self.conf_params["a_max"]
        self.cs_alat_min = self.conf_params["alat_min"]
        self.cs_alat_max = self.conf_params["alat_max"]
        
    #############
    # Callbacks #
    #############
    def _odom_cb(self, data: Odometry) -> None:
        """Odom callback"""
        self.vel_x = data.twist.twist.linear.x
        self.r = data.twist.twist.angular.z
    
    def _odom_frenet_cb(self, data: Odometry) -> None:
        """Odom frenet callback"""
        self.pos_s = data.pose.pose.position.x
        self.pos_n = data.pose.pose.position.y

        self.alpha = data.pose.pose.orientation.z
    
    def _pose_cb(self, data: PoseStamped) -> None:
        """Position callback"""
        self.pos_x = data.pose.position.x
        self.pos_y = data.pose.position.y

        self.theta = euler_from_quaternion([data.pose.orientation.x, data.pose.orientation.y, 
                                    data.pose.orientation.z, data.pose.orientation.w])[2]
    
    def _input_cb(self, data: AckermannDriveStamped) -> None:
        """Steering angle callback. Normally not used since the steering angle of the MPC is taken."""
        
        self.steering_angle = data.drive.steering_angle

    def _imu_cb(self, data: Imu) -> None:
        """acceleration callback. Normally not used."""
        self.acceleration = data.linear_acceleration.x 

    def _global_velocity_scale_cb(self, data: WpntArray) -> None:
        """global_velocity_scale callback. """
        self.glo_v_scale = [x.vx_mps for x in data.wpnts]
    #############
    # Utilities #
    #############
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

    def convert_frenet_to_cartesian(self,frenet_coords)->np.array:
        """
        Converts frenet coordinates to cartesian coordinates.

        Input:  frenet_coords   : Either a np.array or a np.arrray of np.arrays holding the frenet coordinates
        Returns: np.array       : Cartesian coordinates in the shape of the input.
        
        """

        co, si = np.cos(-np.pi/2), np.sin(-np.pi/2)
        R = np.array(((co, -si), (si, co)))

        frenet_coords = np.array(frenet_coords, dtype=np.float32)
        # print("heyo", frenet_coords, frenet_coords.shape)

        if frenet_coords.shape[0] > 2:
            return np.array([self.spline.get_coordinate(s) + d*self.spline.get_derivative(s).reshape(1,2) @ R for s, d in frenet_coords]).reshape(-1,2)                
        
        else:
            
            s = frenet_coords[0]
            d = frenet_coords[1]

            return self.spline.get_coordinate(s) + d*self.spline.get_derivative(s) @ R

    def _dynamics_of_car(self, t, x0)->list:
        '''
        Used for forward propagation. This function takes the dynamics from the acados model.
        '''
    
        s = x0[State.POS_ON_CENTER_LINE_S]
        n = x0[State.MIN_DIST_TO_CENTER_LINE_N]
        alpha = x0[State.ORIENTATION_ALPHA]
        v = x0[State.VELOCITY_V]
        delta = x0[State.STEERING_ANGLE_DELTA]

        derv = x0[5]
        derDelta = x0[6]

        xdot = self.model.f_expl_func(s, n, alpha, v, delta, derv, derDelta, self.model_params.p)

        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]), float(xdot[4]),derv, derDelta]  

        return xdot

    def propagate_time_delay(self, states: np.array, inputs:np.array)->np.array:
        """
        RK45 forward propagation. Calls the function _dynamics_of_car which contains the dynamics of the car.

        Input:  np.array    : Holds the current state which needs to be propagated.
                inputs      : Holds the current inputs from the MPC.

        Returns: np.array   : Propagated state without input. 
        
        """


        # Initial condition on the ODE
        x0 = np.concatenate((states,inputs), axis=0)
        # forward propagation
        solution = solve_ivp(self._dynamics_of_car, t_span= [0, self.t_delay], y0=x0, method='RK45', atol=1e-8, rtol=1e-8)

        solution = [x[-1] for x in solution.y]
        # print("sol", solution)

        # Constraint on max. steering angle
        if abs(solution[State.STEERING_ANGLE_DELTA]) > self.model.delta_max:
            solution[State.STEERING_ANGLE_DELTA] = np.sign(solution[State.STEERING_ANGLE_DELTA]) * self.model.delta_max
        

        # Constraint on max. vel
        if abs(solution[State.VELOCITY_V]) > self.constraint.v_max:
            solution[State.VELOCITY_V] = np.sign(solution[State.VELOCITY_V]) * self.constraint.v_max
        
        # On minimum speed
        if abs(solution[State.VELOCITY_V]) < self.constraint.v_min:
            solution[State.VELOCITY_V] = self.constraint.v_min
        
        
        # Only get the state as solution of where the car will be in t_delay seconds
        # not including the input
        return np.array(solution)[:-2]
    
    def get_initial_position(self, init: bool = False, prev_x0 = None) -> np.ndarray:
        """
        Get updated values for the states from the sensors.
        
        Input:  init     : If controller was initialised, this flag should be set to true
                prev_x0  : Contains the solution state of the MPC for lap wrapping purposes

        Return:  np.array with updated state values
        """
        pos_x = self.pos_x
        pos_y = self.pos_y
        resp_frenet = self.glob2frenet(pos_x, pos_y)
        theta = self.theta
        vel_x = self.vel_x

        if init:
            self.pred_x = np.zeros(self.model.x.size()[0])
            self.steering_angle = 0
            self.acceleration = 0
            prev_x0 = self.pred_x

        alpha = self.alpha
        center_car_s = resp_frenet.s + self.conf_params["lr"]*np.cos(alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = theta - np.arctan2(deriv_center[1],deriv_center[0])
        
        # make aplha_center between -pi and pi
        alpha_center = alpha_center % (2*np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2*np.pi


        delta = self.steering_angle_buf[-1]

        track_length = self.spline.track_length - 0.1 # Needed as values are not exact
        
        # check if the car has completed a lap
        if resp_frenet.s < 0.2 and prev_x0[State.POS_ON_CENTER_LINE_S] // track_length != self.nr_laps:
            # update the number of laps
            self.nr_laps = prev_x0[State.POS_ON_CENTER_LINE_S] // track_length

        current_pos_s = resp_frenet.s + self.nr_laps * self.spline.track_length

        return np.array([current_pos_s, resp_frenet.d, alpha_center, vel_x, delta])
    
    
    def calc_beta(self, x0) -> float:
        """Calc beta according to the approximation used in the model"""
        return -np.arctan(self.conf_params['lr']/(self.conf_params['lr'] + self.conf_params['lf']) * np.tan(x0[4]))
    
    #############
    # Main Loop #
    #############
    def control_loop(self) -> None:
        """Main loop. The control procedure is implemented here."""
        rate = rospy.Rate(1/self.t_MPC)

        x0 = self.get_initial_position(init=True)
        propagated_x = self.propagate_time_delay(x0, np.zeros(2))
        
        # Set the initial state
        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)
        
        # Initinal Solve OCP
        status = self.acados_solver.solve()
        mpc_sd = np.array([self.acados_solver.get(j, "x")[:2] for j in range(self.N+1)])

        # For computation time
        tcomp_max = 0

        while not rospy.is_shutdown():
            
            # start timer
            start = time.perf_counter()

            # update initial state from sensors
            x0 = self.get_initial_position(init=False, prev_x0=x0) # Take D and steering angle delta from MPC solution!

            # time delat compensation
            self.t_delay= self.conp_time+self.t_MPC
            propagated_x = self.propagate_time_delay(x0, np.zeros(2))
            self.publish_propagated_states(propagated_x)
            # visualise the time delay position
            prop_x_plot = self.convert_frenet_to_cartesian(propagated_x[:2])
            self.publish_current_pos(prop_x_plot)

            # Set the new initial state
            self.acados_solver.set(0, "lbx", propagated_x)
            self.acados_solver.set(0, "ubx", propagated_x)

            # dynamically change the target speed, weight parameters and constraints
            np_array_index_online_turn=np.arange(len(self.model_params.p)-13, len(self.model_params.p))

            for i in range(self.N+1):
                idx = np.abs(self.s_ref - mpc_sd[i,0]%self.spline.track_length).argmin()
                traget_v_speed=np.ones(1)*self.glo_v_scale[idx]               
                online_parameters=np.array([traget_v_speed, self.qadv, self.qv, self.qn, self.qalpha,
                                            self.cs_v_min,self.cs_v_max,
                                            self.cs_delta_steer_min,self.cs_delta_steer_max,
                                            self.cs_delta_v_min,self.cs_delta_v_max,
                                            self.cs_alat_min,self.cs_alat_max],dtype=np.float64)
                
                # self.acados_solver.set_params_sparse(i,np_array_index_online_turn, 
                #                                        online_parameters)
                self.acados_solver.set(i,"p",online_parameters)



            # Solve OCP
            status = self.acados_solver.solve()

            # get solution
            x0 = self.acados_solver.get(0, "x")
            self.u0 = self.acados_solver.get(0, "u")
            self.pred_x = self.acados_solver.get(1, "x")
            # steering buffer
            # shift the elements in the buffer to the right and add the predicted steering angle
            self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
            self.steering_angle_buf[0] = self.pred_x[-1]

            # Publish the inputs of the MPC controller
            if status == 0:
                # print(self.pred_x)
                self.publish_ackermann_msg(self.pred_x, True)

                for stage in range(6,1,-1):
                    x = self.acados_solver.get(stage, "x")
                    self.publish_ackermann_msg(x, False)

            ##### visualization #####
            # Creating waypoint array with predicted positions
            mpc_sd = np.array([
                self.acados_solver.get(j, "x")[:2] for j in range(self.N+1)
            ])

            if status == 0:
                states = []
                for node in range(self.N+1):
                    result = self.acados_solver.get(node,"x").tolist()
                    for item in result:
                        states.append(item)
                self.publish_states(states)

            # manage timings
            elapsed = time.perf_counter() - start
            self.conp_time=elapsed
            if elapsed > tcomp_max:
                tcomp_max = elapsed
            self.comp_time_pub.publish(elapsed)
            
            rate.sleep()
    
    ##########################
    # Publishing/Visualizing #
    ##########################
    def publish_current_pos(self, coord: np.array) -> None:
        """Publishes current position into /mpc_controller/current_position"""
        waypoint_marker = Marker()
        waypoint_marker.header.frame_id = "map"
        waypoint_marker.header.stamp = rospy.Time.now()
        waypoint_marker.type = 2
        waypoint_marker.scale.x = 0.2
        waypoint_marker.scale.y = 0.2
        waypoint_marker.scale.z = 0.2
        waypoint_marker.color.r = 0.0
        waypoint_marker.color.g = 1.0
        waypoint_marker.color.b = 0.0
        waypoint_marker.color.a = 1.0
        waypoint_marker.pose.position.x = coord[0]
        waypoint_marker.pose.position.y = coord[1]
        waypoint_marker.pose.position.z = 0
        waypoint_marker.pose.orientation.x = 0
        waypoint_marker.pose.orientation.y = 0
        waypoint_marker.pose.orientation.z = 0
        waypoint_marker.pose.orientation.w = 1
        waypoint_marker.id = 1
        self.pos_pub.publish(waypoint_marker)

    def publish_states(self, states: list) -> None:
        """Publishes states into /mpc_controller/states"""
        msg = Float64MultiArray()
        msg.data = states
        self.states_pub.publish(msg)
    
    def publish_propagated_states(self, states: list) -> None:
        """Publishes propagated state into /mpc_controller/states_propagated"""
        msg = Float64MultiArray()
        msg.data = states
        self.states_propagated_pub.publish(msg)

    def publish_waypoint_markers(self, waypoints: np.ndarray, type:str) -> None:
        """Publishes predicted positions, right, left boundaries and points on reference path."""
        waypoint_markers = MarkerArray()
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = rospy.Time.now()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            if type == "pred":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 1.0
                waypoint_marker.color.a = 1.0
            elif type == "target":
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 1.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0
            else:
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 0.0
                waypoint_marker.color.a = 1.0

            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0
            waypoint_marker.pose.orientation.x = 0
            waypoint_marker.pose.orientation.y = 0
            waypoint_marker.pose.orientation.z = 0
            waypoint_marker.pose.orientation.w = 1
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
            waypoint_markers.markers.append(waypoint_marker)

        if type == "pred":
            self.pred_pos_pub.publish(waypoint_markers)
        elif type == "d_left":
            self.d_left_pub.publish(waypoint_markers)
        elif type == "d_mid":
            self.d_mid_pub.publish(waypoint_markers)
        elif type == "d_right":
            self.d_right_pub.publish(waypoint_markers)
        else:
            self.target_pos_pub.publish(waypoint_markers)

    def publish_current_pos_n(self,state: np.array) -> None:
        """Publishes current distance from reference path in /mpc_controller/current_pos_n. Not used."""
        position = Odometry()
        position.header.stamp = rospy.Time.now()
        position.header.frame_id = "base_link"
        position.pose.pose.position.y = state[State.MIN_DIST_TO_CENTER_LINE_N]
        self.pos_n_pub.publish(position)

    def publish_ackermann_msg(self , state: np.ndarray, next_input: bool = True) -> None:
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.header.frame_id = "base_link"

        ack_msg.drive.steering_angle = state[State.STEERING_ANGLE_DELTA]
        ack_msg.drive.speed = state[State.VELOCITY_V]
        ack_msg.drive.acceleration = 0*self.u0[0]
        # Managing of contol actions sent to controller_node.py
        if next_input:
            # next input
            self.drive_next_pub.publish(ack_msg)
        else:
            # input line
            self.drive_pub.publish(ack_msg)

    def mpc_params_cb(self, params:Config):
        """
        Here the mpc parameters are updated if changed with rqt (dyn reconfigure)
        Values from .yaml file are set in mpc_online_params_server      
        """
        ## mpc Control Parameters
        self.qadv = params.doubles[0].value
        self.qv = params.doubles[1].value
        self.qn = params.doubles[2].value   
        self.qalpha = params.doubles[3].value
        
        # onilne constraints tuning
        self.cs_delta_steer_min = params.doubles[4].value
        self.cs_delta_steer_max = params.doubles[5].value
        self.cs_delta_v_min = params.doubles[6].value
        self.cs_delta_v_max = params.doubles[7].value
        self.cs_v_min = params.doubles[8].value
        self.cs_v_max = params.doubles[9].value
        self.cs_alat_min = params.doubles[10].value
        self.cs_alat_max = params.doubles[11].value
           

if __name__ == "__main__":
    controller = MPCNode()
    controller.control_loop()
