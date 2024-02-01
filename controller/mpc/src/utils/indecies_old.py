from enum import IntEnum

class State(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_ALPHA = 2
    VELOCITY_V = 3
    STEERING_ANGLE_DELTA = 4
    YAW_RATE = 5
    BETA = 6

class Input(IntEnum):
    D_VELOCITY = 0
    D_STEERING_ANGLE = 1


class Parameter(IntEnum):
    m = 0
    C1 = 1
    Cm2 = 2
    Cr0 = 3
    Cr2 = 4
    Cr3 = 5
    Iz = 6
    lr = 7
    lf = 8
    Bf = 9
    Cf = 10
    Df = 11
    Br = 12
    Cr = 13
    Dr = 14
    Imax_c = 15
    Caccel = 16
    Cdecel = 17
    qadv = 18
    qn = 19
    qalpha = 20
    qvy = 21
    freq = 22
    C_S_front = 23
    C_S_rear = 24
    h_cg = 25
    mu = 26
