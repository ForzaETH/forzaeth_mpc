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
    # the input slots of the [x, u] feature vector; dropped by a merge
    # revert in race_stack 94ea48a79, which is why x0[8]/x0[9] were literals
    INPUT_JERK = 8
    INPUT_DSTEERING = 9


class InputIndex(IntEnum):
    D_STEERING_ANGLE = 0
    JERK = 1
