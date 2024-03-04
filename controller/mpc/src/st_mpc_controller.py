#! /usr/bin/env python3

import os
import time
from typing import Tuple

import numpy as np
import rospkg
import rospy
import yaml
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from frenet_conversion.srv import Frenet2Glob, Glob2Frenet
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from st_mpc.acados_settings import acados_settings
from plotting_fnc import *
from scipy.integrate import solve_ivp
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Float64MultiArray
from tf.transformations import euler_from_quaternion
from utils.indecies import Parameter, State
from utils.splinify import SplineTrack
from visualization_msgs.msg import Marker, MarkerArray

from frenet_converter.frenet_converter import FrenetConverter

from steering_lookup.lookup_steer_angle import LookupSteerAngle



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

        LUT_name = rospy.get_param('/LU_table')
        self.steer_lookup = LookupSteerAngle(LUT_name, rospy.loginfo)

        
        self.glob2frenet = rospy.ServiceProxy('convert_glob2frenet_service', Glob2Frenet)
        self.frenet2glob = rospy.ServiceProxy('convert_frenet2glob_service', Frenet2Glob)

        rospy.Subscriber("/car_state/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self._odom_frenet_cb)
        rospy.Subscriber("/car_state/pose", PoseStamped, self._pose_cb)
        rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/input/nav_1", AckermannDriveStamped, self._input_cb)
        rospy.Subscriber("/vesc/sensors/imu/raw", Imu, self._imu_cb)

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
        self.beta_pub = rospy.Publisher("/mpc_controller/beta", Float64, queue_size=10 )


        self._initialize_solver()

    def _init_params(self,) -> None:
        dir_path = rospkg.RosPack().get_path("stack_master")
        self.racecar_version = rospy.get_param("/racecar_version")
        file_path = rospy.get_param("/mpc_config_filename")
        self.conf_file = os.path.join(dir_path, "config", self.racecar_version, file_path)
        with open(self.conf_file, "r") as conf_file:
            self.conf_params = yaml.safe_load(conf_file)
        
        
        self.N = self.conf_params['N']
        self.R = self.conf_params['R']
        
        # Just for cost function testing
        self.track_safety_margin = self.conf_params['track_safety_margin']
        t_delay = self.conf_params['t_delay']
        self.t_MPC = 1 / self.conf_params['MPC_freq']

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
        
        self.spline = SplineTrack(coords_direct=coords_path)
        
        self.nr_laps = 0

        kapparef = [x.kappa_radpm for x in shortest_path.wpnts]
        s0 = self.spline.params
        
        self.constraint, self.model, self.acados_solver, self.model_params = acados_settings(s0, kapparef, d_left, d_right, self.conf_params)

        self.kappa = self.model.kappa

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
        print("final",x0)
        xdot = self.model.f_expl_func(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8], self.model_params.p)

        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]), float(xdot[4]), float(xdot[5]), float(xdot[6]), x0[7], x0[8]]  

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
        print("x0", x0)
        print("I GUESS")
        solution = solve_ivp(self._dynamics_of_car, t_span= [0, self.t_delay], y0=x0, method='RK45', atol=1e-8, rtol=1e-8)
        print("BEFORE")
        solution = [x[-1] for x in solution.y]
        # print("sol", solution)

        # Constraint on max. steering angle
        # if abs(solution[State.STEERING_ANGLE_DELTA]) > self.model.delta_max:
        #     solution[State.STEERING_ANGLE_DELTA] = np.sign(solution[State.STEERING_ANGLE_DELTA]) * self.model.delta_max
        

        # Constraint on max. vel
        # if abs(solution[State.VELOCITY_V]) > self.constraint.v_max:
        #     solution[State.VELOCITY_V] = np.sign(solution[State.VELOCITY_V]) * self.constraint.v_max
        
        # On minimum speed
        # if abs(solution[State.VELOCITY_V]) < self.constraint.v_min:
        #     solution[State.VELOCITY_V] = self.constraint.v_min
        
        
        # Only get the state as solution of where the car will be in t_delay seconds
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
        #my_frenex = self.fren_conv.get_frenet(self.pos_x, self.pos_y)
        #print(f"Error: {resp_frenet.s - my_frenex[0]}, {resp_frenet.d - my_frenex[1]}")
        theta = self.theta
        vel_x = np.clip(self.vel_x, 1, 100)

        if init:
            self.pred_x = np.zeros(self.model.x.size()[0])
            self.steering_angle = 0
            self.acceleration = 0
            prev_x0 = self.pred_x

        # deriv = self.spline.get_derivative(resp_frenet.s)
        # alpha = theta - np.arctan2(deriv[1],deriv[0])
        alpha = self.alpha
        center_car_s = resp_frenet.s + self.conf_params["lr"]*np.cos(alpha)
        deriv_center = self.spline.get_derivative(center_car_s)
        alpha_center = theta - np.arctan2(deriv_center[1],deriv_center[0])

        alpha_center = alpha_center % (2*np.pi)
        if alpha_center > np.pi:
            alpha_center = alpha_center - 2*np.pi


        # delta = self.steering_angle_buf[-1]

        track_length = self.spline.track_length - 0.1 # Needed as values are not exact

        if resp_frenet.s < 0.2 and prev_x0[State.POS_ON_CENTER_LINE_S] // track_length != self.nr_laps:
            self.nr_laps = prev_x0[State.POS_ON_CENTER_LINE_S] // track_length

        current_pos_s = resp_frenet.s + self.nr_laps * self.spline.track_length
        

        return np.array([current_pos_s, resp_frenet.d, vel_x, self.steering_angle, alpha_center, self.r, 0])
    
    def calc_alat(self, x0) -> float:
        return self.conf_params['C1'] * self.pred_x[3] ** 2 \
            + self.u0[0] * np.sin(
                np.arctan(
                    self.conf_params['lr']/(self.conf_params['lr'] + self.conf_params['lf']) \
                    * np.tan(x0[4])
                )
            )
    
    def calc_beta(self, x0) -> float:
        """Calc beta according to the approximation used in the model"""
        return -np.arctan(self.conf_params['lr']/(self.conf_params['lr'] + self.conf_params['lf']) * np.tan(x0[4]))
    
    #############
    # Main Loop #
    #############
    def control_loop(self) -> None:
        print("HALLPPPPPPPPPPPPPPPPPPPPPPPPPPP")
        """Main loop. The control procedure is implemented here."""
        rate = rospy.Rate(1/self.t_MPC)

        x0 = self.get_initial_position(init=True)
        print("HALLLLLLLLLLLLLLLLLLLLLLLLLD")
        propagated_x = self.propagate_time_delay(x0, np.zeros(2))
        # propagated_x = x0

        self.acados_solver.set(0, "lbx", propagated_x)
        self.acados_solver.set(0, "ubx", propagated_x)
        print("HALLKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
        # For plotting
        simX = np.ndarray((int(1e5),self.model.x.size()[0]))
        simU = np.ndarray((int(1e5),self.model.u.size()[0]))
        realX = np.ndarray((int(1e5),self.model.x.size()[0]))
        propX = np.ndarray((int(1e5),self.model.x.size()[0]))
        iter_loop = 1
        simX[0,:] = x0
        realX[0,:] = x0
        propX[0,:] = propagated_x


        # Cost function testing
        advance = []
        deviation = []
        alpha_cost = []
        regularisation = []
        cost_check = []
        cost_acados = []

        # For computation time
        tcomp_sum = 0
        tcomp_max = 0

        while not rospy.is_shutdown():
            print("HALLOOOOOOOOOOOOOOOOOOOOOOOOOO")
            
            start = time.perf_counter()

            # Solve OCP
            status = self.acados_solver.solve()

            if status != 0:
                rospy.logerr(f"acados returned status {status} in closed loop iteration.")

            # get solution
            x0 = self.acados_solver.get(0, "x")
            self.u0 = self.acados_solver.get(0, "u")
            self.pred_x = self.acados_solver.get(1, "x")

            # print(f"A LONG!: {self.u0[0]}")
            a_lat = self.u0[1]
            v = self.pred_x[2]
            st = self.pred_x[3]
            # beta = self.calc_beta(x0)
            # print(f"beta: {}")
            # self.beta_pub.publish(beta)
            # if a_lat > 9 or a_lat < -9:
            #     print(f"A LAT!!!!!!: {a_lat}")

            # TODO calcuate steer with MAP!
            # self.steer = self.steer_lookup.lookup_steer_angle(a_lat, v)

            # getting the cost of each term
            # states = np.array([
            #     self.acados_solver.get(j, "x")[:] for j in range(self.N)
            # ])
            
            # sum_s = 0
            # sum_n = 0
            # sum_alpha = 0
            # sum_regular = 0

            # for idx, state in enumerate(states):
            #     lr = self.model_params.p[Parameter.lr]
            #     lf = self.model_params.p[Parameter.lf]
            #     sdota = (state[State.VELOCITY_V] * np.cos(state[State.ORIENTATION_ALPHA] + np.arctan(lr/(lr + lf) * np.tan(state[State.STEERING_ANGLE_DELTA])))) / (1 - self.kappa(state[State.POS_ON_CENTER_LINE_S]) * state[State.MIN_DIST_TO_CENTER_LINE_N])
            #     sum_s += -self.model_params.p[Parameter.qadv] * sdota * self.t_MPC
            #     if idx == self.N - 1:
            #         sum_n += 5 * self.model_params.p[Parameter.qn] * state[State.MIN_DIST_TO_CENTER_LINE_N]**2
            #         sum_alpha += 5 * self.model_params.p[Parameter.qalpha] * state[State.ORIENTATION_ALPHA]**2   
            #     else:
            #         sum_n += self.model_params.p[Parameter.qn] * state[State.MIN_DIST_TO_CENTER_LINE_N]**2
            #         sum_alpha += self.model_params.p[Parameter.qalpha] * state[State.ORIENTATION_ALPHA]**2
            #         sum_regular += self.u0 @ np.diag(self.R) @ self.u0.reshape(2,1)

            # advance.append(abs(sum_s))
            # deviation.append(sum_n)
            # alpha_cost.append(sum_alpha)
            # regularisation.append(sum_regular[0])
            # cost = sum_s + sum_n + sum_alpha + sum_regular[0]
            # cost_check.append(cost)
            # cost_acados.append(self.acados_solver.get_cost())

            # Publish current state and predicted states
            if status == 0:
                # print(self.pred_x)
                self.publish_ackermann_msg(self.pred_x, True)

                for stage in range(6,1,-1):
                    x = self.acados_solver.get(stage, "x")
                    self.publish_ackermann_msg(x, False)

            # steering buffer
            self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
            self.steering_angle_buf[0] = self.pred_x[-1]
            # print(self.steering_angle_buf)
            
            simX[iter_loop,:] = self.pred_x
            simU[iter_loop,:] = self.u0
            
            # print(iter_loop)

            # Creating waypoint array with predicted positions
            mpc_sd = np.array([
                self.acados_solver.get(j, "x")[:2] for j in range(self.N)
            ])

            if status == 0:
                states = []
                for node in range(self.N):
                    result = self.acados_solver.get(node,"x").tolist()
                    for item in result:
                        states.append(item)
                self.publish_states(states)

            pred_waypoints = self.convert_frenet_to_cartesian(mpc_sd)

            self.publish_waypoint_markers(pred_waypoints, type="pred")


            # time1 = time.perf_counter()
            x0 = self.get_initial_position(init=False, prev_x0=x0) # Take D and steering angle delta from MPC solution!
            # time2 =time.perf_counter()
            # self.beta_pub.publish(np.arctan2(lr,lr + lf) * np.tan(x0[State.STEERING_ANGLE_DELTA]))
        
            propagated_x = self.propagate_time_delay(x0, np.zeros(2))
            # print("propagated", propagated_x)
            self.publish_propagated_states(propagated_x)


            prop_x_plot = self.convert_frenet_to_cartesian(propagated_x[:2])
            self.publish_current_pos(prop_x_plot)

            # Print measured states
            # print(f"s: {x0[0]}, n: {x0[1]}, alpha: {x0[2]}, v: {x0[3]}, delta: {x0[4]}")
            
            realX[iter_loop,:] = x0
            propX[iter_loop,:] = propagated_x
            
            # Set the new initial state
            self.acados_solver.set(0, "lbx", propagated_x)
            self.acados_solver.set(0, "ubx", propagated_x)

            # Print trajectory bounds
            boundaries = np.zeros((self.acados_solver.N + 1, 3, 2))

            for stage in range(self.acados_solver.N + 1):
                x_ = self.acados_solver.get(stage, 'x')
                s_ = x_[State.POS_ON_CENTER_LINE_S]
                s_mod = s_ % self.spline.track_length
                n_ = x_[State.MIN_DIST_TO_CENTER_LINE_N]
                if stage == 1:
                    if self.model.outer_bound_s(s_mod) + n_ <= 0.4 or self.model.inner_bound_s(s_mod) - n_ <= 0.4:
                        rospy.logwarn(f"Outer: {self.model.outer_bound_s(s_mod) + n_}, Inner: {self.model.inner_bound_s(s_mod) - n_}")
                
                boundaries[stage,0,:] = self.convert_frenet_to_cartesian(np.array([s_mod, -self.model.outer_bound_s(s_mod) + self.track_safety_margin]))
                boundaries[stage,2,:] = self.convert_frenet_to_cartesian(np.array([s_mod, self.model.inner_bound_s(s_mod) - self.track_safety_margin]))
                boundaries[stage,1,:] = self.convert_frenet_to_cartesian(np.array([s_mod, 0]))

            self.publish_waypoint_markers(boundaries[:,0,:], "d_left")
            self.publish_waypoint_markers(boundaries[:,1,:], "d_mid")
            self.publish_waypoint_markers(boundaries[:,2,:], "d_right")

            iter_loop += 1

            # manage timings
            elapsed = time.perf_counter() - start
            tcomp_sum += elapsed
            if elapsed > tcomp_max:
                tcomp_max = elapsed
                    
            # print(f"MPC took {elapsed:01.5f} s.")
            self.comp_time_pub.publish(elapsed)
            
            # Plotting
            if iter_loop == 40000:
                simX_plot = simX[:iter_loop, :]
                simU_plot = simU[:iter_loop,:]
                realX_plot = realX[:iter_loop,:]
                print(f"Max. computation time: {tcomp_max}")
                print(f"Average computation time: {tcomp_sum/iter_loop}")

                plot_costs(advance, deviation,alpha_cost, regularisation)
                plot_n(simX_plot, simU_plot, realX_plot)
                plot_alpha(simX_plot, simU_plot, realX_plot)
                plot_vx(simX_plot, simU_plot, realX_plot)

                # plot_res(self.spline, self.mintime_spline, simX_plot,simU_plot, realX_plot)
                # plot_res(self.spline, simX_plot,simU_plot, realX_plot)
                plot_costs()


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

        ack_msg.drive.steering_angle = state[3]
        ack_msg.drive.speed = state[2]

        # Managing of contol actions sent to controller_node.py
        if next_input:
            self.drive_next_pub.publish(ack_msg)
        else:
            self.drive_pub.publish(ack_msg)

if __name__ == "__main__":
    controller = MPCNode()
    controller.control_loop()
