from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np
from frenet_converter.frenet_converter import FrenetConverter
from learning_based_mpc.l4acados.src.l4acados.controllers import \
    ResidualLearningMPC
from learning_based_mpc.l4acados.src.l4acados.models.residual_model import \
    ResidualModel
from learning_based_mpc.mpc_core.ocp.l4a_acados_settings import \
    build_ocp_for_l4acados
from pbl_config import (CarConfig, PacejkaTireConfig, STMPCConfig,
                        TrailingConfig)
from single_track_mpc.utils.indicies import StateIndex
from single_track_mpc.utils.splinify import SplineTrack

try:
    from casadi import DM
    _HAS_DM = True
except Exception:
    _HAS_DM = False
    DM = None


class STMPC_L4A:
    """Drop-in twin of `STMPC` but backed by L4acados (residual dynamics)."""

    def __init__(
        self,
        pose_frenet: np.ndarray,
        waypoints_data: np.ndarray,
        racecar_version: str,
        stmpc_config: STMPCConfig,
        car_config: CarConfig,
        tire_config: PacejkaTireConfig,
        trailing_config: TrailingConfig,
        controller_frequency: float,
        using_gokart: bool,
        residual_model: ResidualModel,
    ) -> None:
        # Store configs
        self.racecar_version = racecar_version
        self.stmpc_config = stmpc_config
        self.car_config = car_config
        self.tire_config = tire_config
        self.trailing_config = trailing_config
        self.controller_frequency = controller_frequency
        self.using_gokart = using_gokart

        # Internal state
        self.t_MPC = 1 / self.stmpc_config.MPC_freq
        self.t_delay = self.stmpc_config.t_delay + self.t_MPC
        self.steering_angle_buf = np.zeros(2)
        self.mpc_sd = np.zeros((self.stmpc_config.N + 1, 2))
        self.u0 = np.zeros(2)
        self.fre_s = 0.0
        self.previous_frenet_s = 0.0
        self.comp_time = 0.0
        self.overtake_d = self.stmpc_config.overtake_d
        self.loop_rate = 40
        self.i_gap = 0

        # Geometry from waypoints
        x = waypoints_data[:, 0]
        y = waypoints_data[:, 1]
        kapparef = waypoints_data[:, 3]
        self.s_ref = waypoints_data[:, 4]
        d_left = waypoints_data[:, 5]
        d_right = waypoints_data[:, 6]

        self.fren_conv = FrenetConverter(x, y)
        coords_path = self.transform_waypoints_to_coords_from_array(waypoints_data)
        self.spline = SplineTrack(coords_direct=coords_path)
        self.nr_laps = 0
        self.track_length = float(self.s_ref[-1]) if len(self.s_ref) else 1.0

        # Nominal OCP (unchanged) + wrap with residual model
        ocp, model, constraint, params = build_ocp_for_l4acados(
            self.s_ref, kapparef, d_left, d_right,
            self.stmpc_config, self.car_config, self.tire_config,
        )
        self.model = model
        self.constraint = constraint
        self.model_params = params

        self.l4a_solver = ResidualLearningMPC(
            ocp=ocp,
            residual_model=residual_model,
        )

        self.set_sim_mu(self.tire_config.friction_coeff)
        self.apply_warm_start(pose_frenet=pose_frenet)

        self.kappa = kapparef
        self.prev_acc = 0.0

    # ======================== Public API (mirror STMPC) =======================
    def main_loop(
        self,
        state: str,
        position_in_map: np.ndarray,
        waypoint_array_in_map: np.ndarray,
        speed_now: float,
        opponent: Union[np.ndarray, None],
        position_in_map_frenet: np.ndarray,
        alpha: float,
        single_track_state: np.ndarray,
    ):
        """Return (speed, acceleration, jerk, steering_angle, states_list, status)."""
        self.state = state
        self.position_in_map = position_in_map
        self.waypoint_array_in_map = waypoint_array_in_map
        self.speed_now = speed_now
        self.opponent = opponent
        self.position_in_map_frenet = position_in_map_frenet
        self.alpha = alpha
        self.vel_y = single_track_state[0]
        self.yaw_rate = single_track_state[1]

        # Current Frenet pose
        self.fre_s = float(self.position_in_map_frenet[0])
        self.fre_d = float(self.position_in_map_frenet[1])
        self.fre_alpha = float(self.alpha)

        # Initial condition (from solver if available)
        try:
            x0 = self.l4a_solver.get(0, "x")
        except Exception:
            x0 = np.zeros(self.model.n_x)
        propagated_x = x0.copy()
        self.l4a_solver.set(0, "lbx", propagated_x)
        self.l4a_solver.set(0, "ubx", propagated_x)

        # Trailing mode
        if (self.state == "TRAILING") and (self.opponent is not None):
            self.trailing_command = self.trailing_controller(self.waypoint_array_in_map[0, 2])
        else:
            self.i_gap = 0

        # Online parameters + bounds per stage
        for i in range(self.stmpc_config.N + 1):
            if (self.state == "TRAILING") and (self.opponent is not None):
                target_v_speed = self.trailing_command
            else:
                idx = np.abs(self.waypoint_array_in_map[:, 4] - (self.mpc_sd[i, 0] % self.track_length)).argmin()
                target_v_speed = self.waypoint_array_in_map[idx, 2]
                self.overtake_d = 0.0
                if (self.state == "OVERTAKE") and (self.opponent is not None):
                    self.overtake_d = self.waypoint_array_in_map[idx, 3]

            if i < self.stmpc_config.steps_delay:
                multiplier = 1e6
                p = np.array([
                    target_v_speed,
                    self.stmpc_config.qadv,
                    self.stmpc_config.qv,
                    self.stmpc_config.qn,
                    self.stmpc_config.qalpha,
                    self.stmpc_config.qjerk,
                    multiplier * self.stmpc_config.qddelta,
                    self.stmpc_config.alat_max,
                    self.stmpc_config.a_min,
                    self.stmpc_config.a_max,
                    self.stmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.stmpc_config.terminal_multiplier,
                ], dtype=np.float64)
            else:
                p = np.array([
                    target_v_speed,
                    self.stmpc_config.qadv,
                    self.stmpc_config.qv,
                    self.stmpc_config.qn,
                    self.stmpc_config.qalpha,
                    self.stmpc_config.qjerk,
                    self.stmpc_config.qddelta,
                    self.stmpc_config.alat_max,
                    self.stmpc_config.a_min,
                    self.stmpc_config.a_max,
                    self.stmpc_config.track_safety_margin,
                    self.overtake_d,
                    self.tire_config.friction_coeff,
                    self.stmpc_config.terminal_multiplier,
                ], dtype=np.float64)

            self.l4a_solver.set(i, "p", p)

            if i < self.stmpc_config.N:
                self.l4a_solver.set(i, "lbu", np.array([-50, self.stmpc_config.ddelta_min]))
                self.l4a_solver.set(i, "ubu", np.array([50, self.stmpc_config.ddelta_max]))
                self.l4a_solver.set(i + 1, "lbx", np.array([
                    self.stmpc_config.v_min,
                    self.stmpc_config.delta_min,
                    self.stmpc_config.a_min,
                ]))
                self.l4a_solver.set(i + 1, "ubx", np.array([
                    self.stmpc_config.v_max,
                    self.stmpc_config.delta_max,
                    self.stmpc_config.a_max,
                ]))

        # Solve
        status = self.l4a_solver.solve()
        if not (status == 0 or status == 2):
            self.apply_warm_start(pose_frenet=[self.fre_s, self.fre_d, self.fre_alpha])
            return 0.0, 0.0, 0.0, getattr(self, "measured_steer", 0.0), None, status

        # Outputs
        self.u0 = self.l4a_solver.get(0, "u")
        self.prev_acc = self.l4a_solver.get(0, "x")[StateIndex.ACCEL.value]
        delayed_index = self.stmpc_config.steps_delay + 1
        self.pred_x = self.l4a_solver.get(delayed_index, "x")

        self.steering_angle = float(self.pred_x[5])
        self.speed = float(self.pred_x[3])
        self.acceleration = float(self.prev_acc + self.u0[0] / self.controller_frequency)
        self.prev_acc = self.acceleration
        self.jerk = float(self.u0[0])

        self.steering_angle_buf[1:] = self.steering_angle_buf[:-1]
        self.steering_angle_buf[0] = self.pred_x[5]
        self.mpc_sd = np.array([self.l4a_solver.get(j, "x")[:2] for j in range(self.stmpc_config.N + 1)])

        states_list = []
        for node in range(self.stmpc_config.N + 1):
            result = self.l4a_solver.get(node, "x").tolist()
            for item in result:
                states_list.append(item)
        return self.speed, self.acceleration, self.jerk, self.steering_angle, states_list, status

    def apply_warm_start(self, pose_frenet: np.ndarray) -> None:
        warm = self.get_warm_start(pose_frenet=pose_frenet, const_acc=1.0, const_steer_vel=0.0)
        for i in range(self.stmpc_config.N + 1):
            self.l4a_solver.set(i, "x", warm[i][: self.model.n_x])
            if i < self.stmpc_config.N:
                self.l4a_solver.set(i, "u", warm[i][self.model.n_x:])

    def get_warm_start(self, pose_frenet: np.ndarray, const_acc: float, const_steer_vel: float) -> np.ndarray:
        x0 = np.zeros(self.model.n_x)
        u0 = np.zeros(self.model.n_u)
        return np.asarray([np.concatenate([x0, u0]) for _ in range(self.stmpc_config.N + 1)])

    # Compatibility shim from early drafts
    def solve(self) -> None:
        self.l4a_solver.solve()

    # =============================== Helpers ================================
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

    # --------------------- Residual GP helper (optional) ---------------------
    @staticmethod
    def make_gp_residual_model(predict_mean: Callable, idx_map: Dict[str, int], nx: int) -> ResidualModel:
        class _GPResidual(ResidualModel):
            def __init__(self, f_mean, idx_map, nx):
                self.f_mean = f_mean
                self.idx_vx = idx_map.get("vx")
                self.idx_vy = idx_map.get("vy")
                self.idx_wz = idx_map.get("wz")
                self.nx = nx

            def eval(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
                vx = x[self.idx_vx]
                vy = x[self.idx_vy]
                wz = x[self.idx_wz]
                r = np.zeros(self.nx)
                dr = np.asarray(self.f_mean(np.array([vx, vy, wz])), dtype=float).reshape(-1)
                r[self.idx_vx] = dr[0]
                if dr.size > 1:
                    r[self.idx_vy] = dr[1]
                if dr.size > 2:
                    r[self.idx_wz] = dr[2]
                return r

            def jac_x(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
                return np.zeros((self.nx, self.nx))

            def jac_u(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
                return np.zeros((self.nx, u.shape[0]))
        return _GPResidual(predict_mean, idx_map, nx)
