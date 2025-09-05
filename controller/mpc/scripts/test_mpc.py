from typing import List, Tuple
from readwrite_global_waypoints import read_global_waypoints
from readwrite_global_waypoints.rw_global_waypoints_src import GlobalWaypoints
from f110_msgs.msg import WpntArray, Wpnt
import numpy as np

# plot raceline, adding arrow at first waypoint for general direction
from plotting_utils import plot_centerline, plot_centerline_with_lr

# plot_centerline(refline_xy)

from pbl_config import CarConfig, PacejkaTireConfig, STMPCConfig, TrailingConfig, load_STMPC_config_ros, load_car_config_ros, load_pacejka_tire_config_ros, load_trailing_config_ros
from single_track_mpc.acados_settings import acados_settings
from single_track_mpc.utils.splinify import SplineTrack
from frenet_converter.frenet_converter import FrenetConverter

# TODO clean up original single_track_mpc.py to isolate Standalone controller and ROS interface
class StandaloneMPCController:
    def __init__(self,
                pose_frenet: np.ndarray,
                waypoints_data: np.ndarray,  # Replace ROS waypoints with numpy array
                racecar_version: str,
                stmpc_config: STMPCConfig,
                car_config: CarConfig,
                tire_config: PacejkaTireConfig,
                trailing_config: TrailingConfig,
                controller_frequency: float,
                using_gokart: bool) -> None:

        # Store configs
        self.racecar_version = racecar_version
        self.stmpc_config = stmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # Initialize parameters
        self.mpc_init_params()

        # Initialize solver with waypoints
        self.mpc_initialize_solver(pose_frenet, waypoints_data)

    def mpc_init_params(self) -> None:
        # MPC parameters
        self.t_MPC = 1 / self.stmpc_config.MPC_freq
        self.t_delay = self.stmpc_config.t_delay + self.t_MPC

        # Steering angle buffer
        buf_size = 2
        self.steering_angle_buf = np.zeros(buf_size)

        # Initial state
        self.mpc_sd = np.zeros((self.stmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0
        self.previous_frenet_s = 0
        self.comp_time = 0
        self.overtake_d = self.stmpc_config.overtake_d
        self.loop_rate = 40

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
        # vx_ref = waypoints_data[:, 2]
        kapparef = waypoints_data[:, 3]
        self.s_ref = waypoints_data[:, 4]
        d_left = waypoints_data[:, 5]
        d_right = waypoints_data[:, 6]

        # Initialize frenet converter
        self.fren_conv = FrenetConverter(x, y)

        # Create spline track
        # coords_path = np.column_stack((x[:-1], y[:-1]))
        # self.spline = SplineTrack(coords_direct=coords_path)

        # Initialize lap counter
        self.nr_laps = 0

        # Setup acados solver
        (self.constraint, self.model, self.acados_solver, self.model_params) = acados_settings(
            self.s_ref, kapparef, d_left, d_right,
            self.stmpc_config, self.car_config, self.tire_config
        )

        # Apply warm start
        self.apply_warm_start(pose_frenet=pose_frenet)

        self.kappa = kapparef
        self.prev_acc = 0

    def apply_warm_start(self, pose_frenet: np.ndarray):
        """Applies a warm start to the MPC solver.

        Args:
            pose_frenet (np.ndarray): _description_
        """
        warm_start_trajectory = self.get_warm_start(pose_frenet=pose_frenet, const_acc=1, const_steer_vel=0.0)
        for i in range(self.stmpc_config.N + 1):
            self.acados_solver.set(i, "x", warm_start_trajectory[i][:self.model.n_x])
            if i < self.stmpc_config.N:
                self.acados_solver.set(i, "u", warm_start_trajectory[i][self.model.n_x:])

    def get_warm_start(self, pose_frenet: np.ndarray, const_acc: float, const_steer_vel: float) -> np.array:
        """
        Returns a warm start trajectory for the MPC. This is done by propagating the current state with a constant acceleration and steering angle.

        Input:  const_acc   : Constant acceleration
                const_steer : Constant steering angle

        Returns: np.array    : Warm start trajectory
        """
        warm_start = np.zeros((self.stmpc_config.N + 1, 10))  # TODO: hardcoded state space dimension
        # TODO setting the velocity to current actual velocity might remove slowing down
        warm_start[0] = np.array([pose_frenet[0], pose_frenet[1], pose_frenet[2],
                                 1, 0, 0, 0, const_acc, 0, const_steer_vel])
        for i in range(1, self.stmpc_config.N + 1):
            xdot = self._dynamics_of_car(0, warm_start[i - 1])
            warm_start[i] = warm_start[i - 1] + np.array(xdot) / self.stmpc_config.MPC_freq  # ugly euler integration
        return warm_start

    def _dynamics_of_car(self, t, x0) -> list:
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
        jerk = x0[9]
        xdot = self.model.f_expl_func(
            s, n, theta, v_x, v_y, delta, yaw_rate, accel, derDelta, jerk, self.model_params.p)

        xdot = [float(xdot[0]), float(xdot[1]), float(xdot[2]), float(xdot[3]),
                float(xdot[4]), float(xdot[5]), float(xdot[6]), float(xdot[7]), derDelta, jerk]

        return xdot


#Usage example:

# Initial pose in frenet coordinates [s, d, alpha]
# initial_pose = np.array([0.0, 0.0, 0.0])

def extract_waypoints_data(waypoints: List[Wpnt]) -> np.ndarray:
    """Convert WpntArray to numpy array for easier processing."""
    waypoints_list = [(wpnt.x_m, wpnt.y_m, wpnt.vx_mps, wpnt.kappa_radpm, wpnt.s_m, wpnt.d_left, wpnt.d_right) for wpnt in waypoints]
    return np.array(waypoints_list)


# Initialize MPC


if __name__ == "__main__":
    # Example of running the MPC solver
    import time

    gb_waypoints: GlobalWaypoints = read_global_waypoints("THETRACK")

    refline: List[Wpnt] = gb_waypoints.global_traj_wpnts_iqp.wpnts

    refline_xy = np.array([(p.x_m, p.y_m) for p in refline])

    left_dist = np.array([p.d_left for p in refline])

    right_dist = np.array([p.d_right for p in refline])

    plot_centerline_with_lr(refline_xy, left_dist, right_dist)


    racecar_version = "NUC5"
    floor = "dubi"
    pose_frenet = np.array([0.0, 0.0, 0.0])  # 0 position on the raceline
    waypoints = extract_waypoints_data(refline)
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)
    car_config: CarConfig = load_car_config_ros(racecar_version)
    tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
    trailing_config = load_trailing_config_ros(racecar_version)

    mpc = StandaloneMPCController(
        pose_frenet=pose_frenet,
        waypoints_data=waypoints,
        racecar_version="f110",
        stmpc_config=stmpc_config,
        car_config=car_config,
        tire_config=tire_config,
        trailing_config=trailing_config,
        controller_frequency=40.0,
        using_gokart=False
    )

    start_time = time.time()
    for _ in range(10):  # Run for 10 iterations as an example
        mpc.acados_solver.solve()
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        start_time = time.time()
        # Retrieve and print the optimal control inputs
        u_opt = mpc.acados_solver.get(0, "u")
        print(f"Optimal control input: {u_opt}")
