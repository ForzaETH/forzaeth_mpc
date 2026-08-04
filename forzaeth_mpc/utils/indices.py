from enum import IntEnum

class StateIndex(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_THETA = 2
    VELOCITY_V_X = 3
    VELOCITY_V_Y = 4
    STEERING_ANGLE_DELTA = 5
    YAW_RATE = 6
    ACCEL = 7

class Input(IntEnum):
    JERK = 0
    D_STEERING_ANGLE = 1
