"""State and input layout of the single-track model.

One module on purpose. There used to be a second copy under ``lbmpc/utils/indicies.py``
whose same-named members held transposed values, so importing the wrong one was a silent
wrong answer rather than an ImportError. ``research/`` keeps its own, deliberately, for
the frozen pre-unification layout it is pinned to.

The order is the model's: see the probe in ``stmpc/bicycle_model.py``, which asserts it
against ``f_expl`` rather than trusting this file.
"""
from enum import IntEnum


class StateIndex(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_THETA = 2
    VELOCITY_V_X = 3
    VELOCITY_V_Y = 4
    YAW_RATE = 5
    STEERING_ANGLE_DELTA = 6
    ACCEL = 7
    # the input slots of the [x, u] feature vector; dropped by a merge revert in
    # race_stack 94ea48a79, which is why x0[8]/x0[9] used to be literals
    INPUT_JERK = 8
    INPUT_DSTEERING = 9


class InputIndex(IntEnum):
    D_STEERING_ANGLE = 0
    JERK = 1
