from typing import Dict, Tuple

import numpy as np
from single_track_mpc.STMPC import STMPC


class ControllerMPC:
    """Thin façade over STMPC/ACADOS: set params, pin x0, solve, read u/cost."""

    def __init__(self, mpc: STMPC, track_len: float):
        self.mpc = mpc
        self.H = mpc.stmpc_config.N
        self.nx = mpc.model.n_x
        # Try spline first; if missing, we’ll fallback to waypoint-based track_len upstream.
        # self.track_len = float(getattr(getattr(mpc, "spline", None), "track_length", 0.0))
        self.track_len = track_len

    def set_online_params(self,
                          waypoints: np.ndarray,
                          qdict: Dict[str, float],
                          mpc_mu: float,
                          track_len_fallback: float):
        track_len = self.track_len if self.track_len > 0.0 else float(track_len_fallback)

        for i in range(self.H + 1):
            # reference s along horizon
            try:
                s_ref = float(self.mpc.mpc_sd[i, 0]) % track_len
            except Exception:
                s_ref = float(waypoints[min(i, len(waypoints) - 1), 4]) % track_len

            idx = int(np.abs(waypoints[:, 4] - s_ref).argmin())
            target_v = float(waypoints[idx, 2])

            # p vector — matches your current model layout (includes terminal_multiplier)
            p = np.array([
                target_v,
                qdict["q_adv"], qdict["q_v"], qdict["q_n"], qdict["q_alpha"],
                qdict["q_jerk"], qdict["q_ddelta"],
                self.mpc.stmpc_config.alat_max,
                self.mpc.stmpc_config.a_min,
                self.mpc.stmpc_config.a_max,
                self.mpc.stmpc_config.track_safety_margin,
                0.0,
                float(mpc_mu),
                qdict["terminal_multiplier"]
            ], dtype=np.float64)
            try:
                self.mpc.acados_solver.set(i, "p", p)
            except Exception:
                print(f"Setting acados parameters not succesful!")
                # If your compiled ocp expects a different p length/order, adjust here.
                pass

    def pin_state(self, x0_full: np.ndarray):
        x0 = np.asarray(x0_full[:self.nx], dtype=float)
        self.mpc.acados_solver.set(0, "lbx", x0)
        self.mpc.acados_solver.set(0, "ubx", x0)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        status = int(self.mpc.acados_solver.solve())
        if status not in (0, 2):
            try:
                self.mpc.apply_warm_start(pose_frenet=self.mpc.pose_frenet)
            except Exception:
                print(f"Applying warm start after solver not converging not succesful!")
                pass

        try:
            cost = float(self.mpc.acados_solver.get_cost())
        except Exception:
            print(f"Didn't recieve the cost of acados solver")
            cost = float("nan")

        u0 = np.asarray(self.mpc.acados_solver.get(0, "u"), dtype=float).copy()

        # keep mpc_sd rolling for the next step
        try:
            self.mpc.mpc_sd = np.array(
                [self.mpc.acados_solver.get(j, "x")[:2] for j in range(self.H + 1)]
            )
        except Exception:
            print(f"Keep mpc_sd for next step not succesful!")
            pass

        return u0, self.mpc.mpc_sd.copy(), cost, status
