"""Single source of truth for the acados online-parameter (p) vector layout.

The layout is defined by ``bicycle_model()`` in ``bicycle_model.py``:

    0  V_target             7  a_lat_max          14 Bf
    1  weight_adv           8  a_long_min         15 Cf
    2  weight_qv            9  a_long_max         16 Df
    3  weight_qn           10  safety_margin      17 Ef
    4  weight_qalpha       11  overtake_d         18 Br
    5  weight_qjerk        12  MU                 19 Cr
    6  weight_ddelta       13  terminal_multiplier 20 Dr
                                                  21 Er

Anything that calls ``acados_solver.set(i, "p", ...)`` must build it here.
"""

import numpy as np

from forzaeth_mpc.config import PacejkaTireConfig, STMPCConfig

WEIGHT_KEYS = ("q_adv", "q_v", "q_n", "q_alpha", "q_jerk", "q_ddelta", "terminal_multiplier")


def build_online_params(stmpc_config: STMPCConfig,
                        tire_config: PacejkaTireConfig,
                        target_v: float,
                        overtake_d: float,
                        mu: float,
                        weights: dict = None) -> np.ndarray:
    """Builds the 22-entry acados online parameter vector.

    Args:
        stmpc_config: static MPC config (cost weights, accel/margin limits).
        tire_config: Pacejka coefficients.
        target_v: target longitudinal speed for this stage.
        overtake_d: lateral reference offset for this stage.
        mu: friction coefficient handed to the MPC model.
        weights: optional per-call overrides of the cost weights, keys out of
            ``WEIGHT_KEYS``; missing keys fall back to ``stmpc_config``.
    """
    w = {
        "q_adv": stmpc_config.qadv,
        "q_v": stmpc_config.qv,
        "q_n": stmpc_config.qn,
        "q_alpha": stmpc_config.qalpha,
        "q_jerk": stmpc_config.qjerk,
        "q_ddelta": stmpc_config.qddelta,
        "terminal_multiplier": stmpc_config.terminal_multiplier,
    }
    if weights:
        w.update({k: weights[k] for k in WEIGHT_KEYS if k in weights})

    return np.array([
        target_v,
        w["q_adv"],
        w["q_v"],
        w["q_n"],
        w["q_alpha"],
        w["q_jerk"],
        w["q_ddelta"],
        stmpc_config.alat_max,
        stmpc_config.a_min,
        stmpc_config.a_max,
        stmpc_config.track_safety_margin,
        overtake_d,
        mu,
        w["terminal_multiplier"],
        tire_config.Bf,
        tire_config.Cf,
        tire_config.Df,
        tire_config.Ef,
        tire_config.Br,
        tire_config.Cr,
        tire_config.Dr,
        tire_config.Er,
    ], dtype=np.float64)


if __name__ == "__main__":
    from forzaeth_mpc.config.PacejkaTireConfig import \
        load_pacejka_tire_config_ros
    from forzaeth_mpc.config.STMPCConfig import load_STMPC_config_ros

    stmpc_config = load_STMPC_config_ros("NUC5")
    tire_config = load_pacejka_tire_config_ros("NUC5", "dubi")

    p = build_online_params(stmpc_config, tire_config, target_v=3.5, overtake_d=0.25, mu=0.63)
    assert len(p) == 22, len(p)
    assert p[0] == 3.5
    assert p[6] == stmpc_config.qddelta
    assert p[11] == 0.25
    assert p[12] == 0.63, "index 12 must be the friction coefficient"
    assert p[13] == stmpc_config.terminal_multiplier
    assert p[14] == tire_config.Bf, "index 14 must be Bf"
    assert p[21] == tire_config.Er

    p_override = build_online_params(stmpc_config, tire_config, 3.5, 0.0, 0.63,
                                     weights={"q_ddelta": 1e6 * stmpc_config.qddelta})
    assert p_override[6] == 1e6 * stmpc_config.qddelta
    assert np.array_equal(np.delete(p_override, [6, 11]), np.delete(p, [6, 11]))

    print(f"OK: p layout is 22 entries, mu@12={p[12]}, Bf@14={p[14]}")
