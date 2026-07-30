from typing import Dict, Tuple

import numpy as np
from forzaeth_mpc.stmpc.online_params import build_online_params
from forzaeth_mpc.stmpc.STMPC import STMPC


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
            # reference s along horizon (mpc_sd is always (N+1, 2), see mpc_init_params)
            s_ref = float(self.mpc.mpc_sd[i, 0]) % track_len

            idx = int(np.abs(waypoints[:, 4] - s_ref).argmin())
            target_v = float(waypoints[idx, 2])

            p = build_online_params(self.mpc.stmpc_config, self.mpc.tire_config,
                                    target_v=target_v,
                                    overtake_d=0.0,
                                    mu=float(mpc_mu),
                                    weights=qdict)
            self.mpc.acados_solver.set(i, "p", p)

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
