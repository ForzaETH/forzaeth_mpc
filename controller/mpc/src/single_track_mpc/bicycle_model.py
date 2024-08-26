#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import types
# author: Daniel Kloeser
from math import sqrt

import casadi as cs
import numpy as np
from casadi import MX, Function, cos, fmod, if_else, interpolant, sin, vertcat
from nav_msgs.msg import Odometry


def bicycle_model(s0: list, kapparef: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[:length - 1] - s0[length - 1]], s0)
    kapparef = np.append(kapparef, kapparef[1:length])
    kapparef = np.append([kapparef[:length - 1] - kapparef[length - 1]], kapparef)

    d_left = np.append(d_left, d_left[1:length])
    d_left = np.append([d_left[:length - 1] - d_left[length - 1]], d_left)
    d_right = np.append(d_right, d_right[1:length])
    d_right = np.append([d_right[:length - 1] - d_right[length - 1]], d_right)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    # CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    theta = MX.sym("theta")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    delta = MX.sym("delta")
    yaw_rate = MX.sym("yaw_rate")
    x = vertcat(s, n, theta, v_x, v_y, delta, yaw_rate)

    # controls
    accel = MX.sym("accel")
    derDelta = MX.sym("derDelta")
    u = vertcat(accel, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    thetadot = MX.sym("thetadot")
    v_x_dot = MX.sym("v_x_dot")
    v_y_dot = MX.sym("v_y_dot")
    deltadot = MX.sym("deltadot")
    yaw_ratedot = MX.sym("yaw_ratedot")

    xdot = vertcat(sdot, ndot, thetadot, v_x_dot, v_y_dot, deltadot, yaw_ratedot)

    # algebraic variables
    z = vertcat([])

    # parameters
    lr = cfg_dict['lr']
    lf = cfg_dict['lf']
    freq = cfg_dict['MPC_freq']

    m = cfg_dict['m']
    Iz = cfg_dict['Iz']
    lf = cfg_dict['lf']
    lr = cfg_dict['lr']
    Bf = cfg_dict['Bf']  # tire parameters
    Cf = cfg_dict['Cf']
    Df = cfg_dict['Df']
    Ef = cfg_dict['Ef']
    Br = cfg_dict['Br']
    Cr = cfg_dict['Cr']
    Dr = cfg_dict['Dr']
    Er = cfg_dict['Er']
    h_cg = cfg_dict['h_cg']
    g = 9.81
    friction_coeff = cfg_dict['friction_coeff']

    # # online parameters
    V_target = MX.sym('V_target')
    weight_adv = MX.sym('weight_adv')
    weight_qn = MX.sym('weight_qn')
    weight_qv = MX.sym('weight_qv')
    weight_qalpha = MX.sym('weight_qalpha')
    weight_qac = MX.sym('weight_qac')
    weight_ddelta = MX.sym('weight_ddelta')
    cs_a_lat_min = MX.sym('cs_a_lat_min')
    cs_a_lat_max = MX.sym('cs_a_lat_max')
    p = vertcat(V_target, weight_adv, weight_qv, weight_qn, weight_qalpha, weight_qac, weight_ddelta,  # weights
                cs_a_lat_min, cs_a_lat_max)  # lateral acceleration constraints

    # calculate the index
    s_mod = fmod(s, pathlength)

    # single track model definition with pacejka tire model
    # tire model
    # side slip angle
    v_0 = 0.1
    func_v = v_x * (1 / (1 + cs.exp(-v_x / v_0))) - v_x * \
        (1 / (1 + cs.exp(v_x / v_0))) + v_0 * (1 / (cs.cosh(v_x / v_0)))
    alpha_f = (cs.atan2((-v_y - (lf * yaw_rate)), v_x + v_0) + delta)
    alpha_r = (cs.atan2((-v_y + (lr * yaw_rate)), v_x + v_0))
    # load transfer
    # F_zf = m * ((-accel*h_cg)+(g*lr))/(lf+lr)
    # F_zr = m * ((accel*h_cg)+(g*lf))/(lf+lr)
    F_zf = m * g * lr / (lf + lr)
    F_zr = m * g * lf / (lf + lr)
    # lateral tire forces
    F_yf = friction_coeff * Df * F_zf * sin(Cf * cs.atan(Bf * alpha_f - Ef * (Bf * alpha_f - cs.atan(Bf * alpha_f))))
    F_yr = friction_coeff * Dr * F_zr * sin(Cr * cs.atan(Br * alpha_r - Er * (Br * alpha_r - cs.atan(Br * alpha_r))))

    # motor force
    F_motor = accel*m
    motor_split_front = 0.5
    motor_split_rear = 1 - motor_split_front
    F_xf = motor_split_front*F_motor
    F_xr = motor_split_rear*F_motor

    # dynamics
    sdota = ((v_x*cos(theta)) - (v_y*sin(theta))) / (1 - kapparef_s(s_mod) * n)
    ndot = v_x*sin(theta) + v_y*cos(theta)
    der_theta = yaw_rate - (kapparef_s(s_mod)*(((v_x*cos(theta)) - (v_y*sin(theta))) / (1 - kapparef_s(s_mod) * n)))
    yaw_ratedot = (1/Iz)*((-F_yr*lr)+(F_yf*lf)*cos(delta))
    der_v_x = (1/m) * (
        F_xr
        + F_xf * cos(delta)
        - F_yf * sin(delta)
        + m * v_y * yaw_rate
    )
    der_v_y = (1/m) * (
        F_yr
        + F_xf * sin(delta)
        + F_yf * cos(delta)
        - m * v_x * yaw_rate
    )
    der_delta = (derDelta)

    # continuous dynamics
    f_expl = vertcat(
        sdota,
        ndot,
        der_theta,
        der_v_x,
        der_v_y,
        der_delta,
        yaw_ratedot,
    )
    f_expl_func = Function('f_expl_func', [s, n, theta, v_x, v_y, delta, yaw_rate, accel, derDelta, p], [f_expl])

    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 1, 0, 0, 0])

    # model.cost_expr_ext_cost =  -weight_adv*sdota/freq + weight_qn * n**2 + weight_qalpha * theta**2 +weight_qv*(v_x-V_target)**2 +weight_qac*accel**2+weight_ddelta*derDelta**2
    # model.cost_expr_ext_cost_e = -weight_adv*sdota/freq + weight_qn * n**2 + weight_qalpha * theta**2 +weight_qv*(v_x-V_target)**2

    model.cost_expr_ext_cost_0 = weight_qn * n**2 + weight_qalpha * theta**2 + weight_qv * \
        (v_x - V_target)**2 + weight_qac * accel**2 + weight_ddelta * derDelta**2
    model.cost_expr_ext_cost = weight_qn * n**2 + weight_qalpha * theta**2 + weight_qv * \
        (v_x - V_target)**2 + weight_qac * accel**2 + weight_ddelta * derDelta**2
    model.cost_expr_ext_cost_e = 100 * (weight_qn * n**2 + weight_qalpha * theta**2 + weight_qv * (v_x - V_target)**2)
    if cfg_dict["vy_minimization"]:
        model.cost_expr_ext_cost_0 += 100 * v_y**2
        model.cost_expr_ext_cost += 100 * v_y**2
        model.cost_expr_ext_cost_e += 100 * (100 * v_y**2)

    # constraint on lateral errors
    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n
    pos_s = MX.sym('pos_s')
    pos_n = MX.sym('pos_n')
    model.n_outer_bound = Function('outer_bound', [pos_s, pos_n], [outer_bound_s(pos_s) + pos_n])
    model.n_inner_bound = Function('inner_bound', [pos_s, pos_n], [inner_bound_s(pos_s) - pos_n])

    # nonlinear constraints
    a_lat = v_x * v_x * cs.tan(delta) / (lr + lf)
    constraint.pathlength = pathlength
    # constraint.expr = vertcat(n_inner_bound, n_outer_bound)
    ellipse_acc_constraint=(a_lat/cs_a_lat_max)**2+(der_v_x/(10*weight_ddelta))**2
    zero=0
    # long_acceleration=der_v_x
    # dynamic_turn_constraint_min=vertcat(a_lat-cs_a_lat_min)
    # dynamic_turn_constraint_max=vertcat(a_lat-cs_a_lat_max)
    constraint.expr = vertcat(zero, n_inner_bound, n_outer_bound, ellipse_acc_constraint)
    constraint.expr_e = vertcat(n_inner_bound, n_outer_bound)


    ### read the parameters range from the yaml file
    # Model bounds
    model.n_min = cfg_dict['track_safety_margin']  # width of the track [m]
    model.n_max = cfg_dict['track_max_width']  # width of the track [m]

    # state bounds
    model.delta_min = cfg_dict["delta_min"]  # minimum steering angle [rad]
    model.delta_max = cfg_dict["delta_max"]  # maximum steering angle [rad]
    constraint.v_x_min = cfg_dict["v_min"]  # minimum velocity [m/s]
    constraint.v_x_max = cfg_dict["v_max"]  # maximum velocity [m/s]
    model.a_min = cfg_dict["a_min"]  # minimum velocity change rate
    model.a_max = cfg_dict["a_max"]  # maximum velocity change rate

    # input bounds
    model.ddelta_min = cfg_dict["ddelta_min"]  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = cfg_dict["ddelta_max"]  # maximum change rate of steering angle [rad/s]

    # Define model struct
    params = types.SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    model.kappa = kapparef_s
    model.f_expl_func = f_expl_func
    model.outer_bound_s = outer_bound_s
    model.inner_bound_s = inner_bound_s
    return model, constraint, params
