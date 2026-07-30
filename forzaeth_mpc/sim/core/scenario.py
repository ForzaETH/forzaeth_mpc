from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from forzaeth_mpc.config import (load_car_config_ros, load_pacejka_tire_config_ros,
                        load_STMPC_config_ros, load_trailing_config_ros)
from forzaeth_mpc.sim.tracks import load_track
from forzaeth_mpc.stmpc.STMPC import STMPC


class Scenario:
    """
    Loads a track + configs and can build a ready-to-use STMPC object.

    - waypoints: canonical (N,7) array
    - track_len: float
    - make_mpc(dt): builds your STMPC using your real constructor
    - make_x0_full(mpc, s0,n0,psi0,vx0): returns a robust x0_full vector
    """

    def __init__(self,
                 track: str,
                 racecar_version: str,
                 floor: str,
                 refline: str = "iqp"):
        self.track = track
        self.racecar_version = racecar_version
        self.floor = floor
        self.refline_kind = refline

        self.waypoints, self.track_len = load_track(track, refline=refline)

        # configs (same as in your scripts)
        self.stmpc_config = load_STMPC_config_ros(racecar_version)
        self.car_config = load_car_config_ros(racecar_version)
        self.tire_config = load_pacejka_tire_config_ros(racecar_version, floor)
        self.trailing_config = load_trailing_config_ros(racecar_version)

    # ---- Builders -------------------------------------------------------- #
    def make_mpc(self,
                 s0: float,
                 n0: float,
                 psi0: float,
                 dt: float):
        pose_frenet = np.array([s0, n0, psi0], dtype=float)
        mpc = STMPC(
            pose_frenet=pose_frenet,
            waypoints_data=self.waypoints,
            racecar_version=self.racecar_version,
            stmpc_config=self.stmpc_config,
            car_config=self.car_config,
            tire_config=self.tire_config,
            trailing_config=self.trailing_config,
            controller_frequency=(1.0 / dt),
            using_gokart=False
        )
        return mpc

    def make_x0_full(self,
                     mpc: STMPC,
                     s0: float, n0: float, psi0: float, vx0: float) -> np.ndarray:
        nx_full = int(getattr(mpc.model, "n_x_full", 10))
        x0 = np.zeros((nx_full,), dtype=float)
        x0[0], x0[1], x0[2], x0[3] = s0, n0, psi0, vx0
        return x0
