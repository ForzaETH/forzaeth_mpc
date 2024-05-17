
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

# author: Daniel Kloeser

import types

import casadi as cs
import numpy as np
from casadi import MX, Function, cos, interpolant, mod, sin, vertcat


def bicycle_model(s0: list, kapparef: list,vx_ref: list ,d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end 
    # Something wrong：s0[1:length]
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[:length -1 ] - s0[length - 1]], s0)

    kapparef = np.append(kapparef, kapparef[1:length])
    kapparef = np.append([kapparef[:length - 1] - kapparef[length - 1]], kapparef)

    vx_ref = np.append(vx_ref, vx_ref[1:length])
    vx_ref = np.append([vx_ref[:length - 1] - vx_ref[length - 1]], vx_ref)

    d_left = np.append(d_left, d_left[1:length])
    d_left = np.append([d_left[:length -1 ] - d_left[length - 1]], d_left)

    d_right = np.append(d_right, d_right[1:length])
    d_right = np.append([d_right[:length -1 ] - d_right[length - 1]], d_right)


    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    vx_ref_s = interpolant("vx_ref_s", "bspline", [s0], vx_ref)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    delta = MX.sym("delta")
    yaw_rate = MX.sym("yaw_rate")
    accel = MX.sym("accel")

    x = vertcat(s, n, alpha, v_x, v_y, delta, yaw_rate, accel)

    # controls
    jerk = MX.sym("jerk")
    derDelta = MX.sym("derDelta")
    u = vertcat(jerk, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    thetadot = MX.sym("thetadot")
    v_x_dot = MX.sym("v_x_dot")
    v_y_dot = MX.sym("v_y_dot")
    deltadot = MX.sym("deltadot")
    yaw_ratedot = MX.sym("yaw_ratedot")
    accel_dot = MX.sym("accel_dot")
    xdot = vertcat(sdot, ndot, thetadot, v_x_dot, v_y_dot, deltadot, yaw_ratedot, accel_dot)


    # algebraic variables
    z = vertcat([])

    # parameters
    R = np.diag(cfg_dict['R'])
    lr = cfg_dict['lr']
    lf = cfg_dict['lf']
    freq = cfg_dict['MPC_freq']

    m = cfg_dict['m']
    Iz = cfg_dict['Iz']
    lf = cfg_dict['lf']
    lr = cfg_dict['lr']
    Bf = cfg_dict['Bf']#tire parameters
    Cf = cfg_dict['Cf']
    Df = cfg_dict['Df']
    Ef = cfg_dict['Ef'] 
    Br = cfg_dict['Br']
    Cr = cfg_dict['Cr']
    Dr = cfg_dict['Dr']
    Er = cfg_dict['Er']
    h_cg = cfg_dict['h_cg']
    g=9.81
    friction_coeff = cfg_dict['friction_coeff']


    # parameters
    V_target = MX.sym('V_target')
    weight_adv = MX.sym('weight_adv')
    weight_qn = MX.sym('weight_qn')
    weight_qv = MX.sym('weight_qv')
    weight_qalpha = MX.sym('weight_qalpha')
    cs_vx_min = MX.sym('cs_vx_min')
    cs_vx_max = MX.sym('cs_vx_max')
    cs_delta_steer_min = MX.sym('cs_delta_steer_min')
    cs_delta_steer_max = MX.sym('cs_delta_steer_max')
    cs_delta_v_min = MX.sym('cs_delta_v_min')
    cs_delta_v_max = MX.sym('cs_delta_v_max')
    cs_a_lat_min = MX.sym('cs_a_lat_min')
    cs_a_lat_max = MX.sym('cs_a_lat_max')
    p = vertcat(V_target,weight_adv,weight_qv,weight_qn,weight_qalpha, # weights
                cs_vx_min,cs_vx_max,    # velocity constraints
                cs_delta_steer_min,cs_delta_steer_max,  # steering angle constraints
                cs_delta_v_min,cs_delta_v_max,   # velocity change rate constraints
                cs_a_lat_min, cs_a_lat_max) # lateral acceleration constraints
    
    # calculate the index
    s_mod = mod(s, pathlength)

    # single track model definition with pacejka tire model

    ## tire model
    #side slip angle
    v_0 = 1.0
    func_v = v_x*(1/(1 + cs.exp(-v_x/v_0))) - v_x*(1/(1 + cs.exp(v_x/v_0))) + v_0*(1/(cs.cosh(v_x/v_0)))
    alpha_f = (cs.atan2((-v_y - (lf*yaw_rate)),func_v) + delta)
    alpha_r = (cs.atan2((-v_y + (lr*yaw_rate)),func_v)) 
    # load transfer
    F_zf = m * ((-accel*h_cg)+(g*lr))/(lf+lr)
    F_zr = m * ((accel*h_cg)+(g*lf))/(lf+lr)
    # lateral tire forces
    F_yf = friction_coeff*Df*F_zf*sin(Cf*cs.atan(Bf*alpha_f - Ef*(Bf*alpha_f - cs.atan(Bf*alpha_f))))
    F_yr = friction_coeff*Dr*F_zr*sin(Cr*cs.atan(Br*alpha_r - Er*(Br*alpha_r - cs.atan(Br*alpha_r))))

    dot_s = ((v_x*cos(alpha)) - (v_y*sin(alpha))) / (1 - kapparef_s(s_mod) * n)
    dot_n = v_x*sin(alpha) + v_y*cos(alpha) 
    dot_alpha = yaw_rate - (kapparef_s(s_mod)*(((v_x*cos(alpha)) - (v_y*sin(alpha))) / (1 - kapparef_s(s_mod) * n)))
    dot_yawrate = (1/Iz)*((-F_yr*lr)+(F_yf*lf)*cos(delta))
    dot_vx = accel +((1/m)*(-F_yf*sin(delta))) + (v_y*yaw_rate) 
    dot_vy = ((1/m)*(F_yr+F_yf*cos(delta))) - (v_x*yaw_rate)

    # continuous dynamics 
    f_expl = vertcat(
        dot_s,
        dot_n,
        dot_alpha,
        dot_vx,
        dot_vy,
        derDelta,
        dot_yawrate,
        jerk
    )
    f_expl_func = Function('f_expl_func', [s, n, alpha, v_x, v_y, delta, yaw_rate, accel, jerk, derDelta, p], [f_expl])

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 1, 0, 0, 0, 0])

    # Definition of external cost function
    model.cost_expr_ext_cost =  -weight_adv*dot_s**2/freq+weight_qn * n**2 + weight_qalpha * alpha**2 +weight_qv*(v_x-V_target)**2+u.T @ R @ u
    model.cost_expr_ext_cost_e = -weight_adv*dot_s**2/freq+weight_qn * n**2 + weight_qalpha * alpha**2+weight_qv*(v_x-V_target)**2

    # constraint on forces
    a_lat = 0

    # constraint on lateral errors
    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n 

    pos_s = MX.sym('pos_s')
    pos_n = MX.sym('pos_n')
    model.n_outer_bound = Function('outer_bound', [pos_s,pos_n], [outer_bound_s(pos_s) + pos_n])
    model.n_inner_bound = Function('inner_bound', [pos_s,pos_n], [inner_bound_s(pos_s) - pos_n])

    #  nonlinear constraints
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    # construct online parameter turning constraints
    # e.g. 0<v-cs_vx_min<inf, -inf<v-cs_vx_max<0 equivalent to v>cs_vx_min, v<cs_vx_max
    dynamic_turn_constraint_min=vertcat(v_x-cs_vx_min,accel-cs_delta_v_min,derDelta-cs_delta_steer_min,a_lat-cs_a_lat_min)
    dynamic_turn_constraint_max=vertcat(v_x-cs_vx_max,accel-cs_delta_v_max,derDelta-cs_delta_steer_max,a_lat-cs_a_lat_max)
    constraint.expr = vertcat(dynamic_turn_constraint_min, dynamic_turn_constraint_max, n_inner_bound, n_outer_bound)

    ### read the parameters range from the yaml file
    # Model bounds
    model.n_min = cfg_dict['track_safety_margin']  # width of the track [m]
    model.n_max = cfg_dict['track_max_width'] # width of the track [m]
    # state bounds
    model.delta_min = cfg_dict["delta_min"] # minimum steering angle [rad]
    model.delta_max = cfg_dict["delta_max"] # maximum steering angle [rad]
    model.dv_min = cfg_dict["a_min"] # minimum velocity change rate
    model.dv_max = cfg_dict["a_max"] # maximum velocity change rate

    # input bounds
    model.ddelta_min = cfg_dict["ddelta_min"] # minimum change rate of stering angle [rad/s]
    model.ddelta_max = cfg_dict["ddelta_max"] # maximum change rate of steering angle [rad/s]
    model.jerk_min = cfg_dict["jerk_min"] # minimum dot velocity change rate
    model.jerk_max = cfg_dict["jerk_max"] # maximum dot velocity change rate


    # nonlinear constraint
    constraint.alat_min = cfg_dict["alat_min"] # maximum lateral force [m/s^2]
    constraint.alat_max = cfg_dict["alat_max"] # maximum lateral force [m/s^1]

    # Velocity constraint
    constraint.v_min = cfg_dict["v_min"] # minimum velocity [m/s]
    constraint.v_max = cfg_dict["v_max"] # maximum velocity [m/s]



    # Define model struct
    params = types.SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.dyn_disc_fun = 0
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

