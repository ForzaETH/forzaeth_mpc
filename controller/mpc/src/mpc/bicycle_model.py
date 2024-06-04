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

# author: Daniel Kloeser

import types

import casadi as cs
import numpy as np
from casadi import MX, Function, cos, interpolant, fmod, sin, vertcat


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



    # N = cfg_dict['N']
    # Tf = cfg_dict['Tf']


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
    v = MX.sym("v")
    delta = MX.sym("delta")

    x = vertcat(s, n, alpha, v, delta)

    # controls
    der_v = MX.sym("der_v")
    derDelta = MX.sym("derDelta")
    u = vertcat(der_v, derDelta)

    # Delta formulation
    # next_v = v + der_v/N
    # next_delta = delta + (Tf)*derDelta/N 

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, deltadot)

    # algebraic variables
    z = vertcat([])

    # parameters
    m = MX.sym("m")
    C1 = MX.sym("C1")
    C2 = MX.sym("C2")
    Cr0 = MX.sym("Cr0")
    Cr2 = MX.sym("Cr2")
    Cr3 = MX.sym("Cr3")
    Iz = MX.sym("Iz")
    lr = MX.sym("lr")
    lf = MX.sym("lf")
    Df = MX.sym("Df")
    Cf = MX.sym("Cf")
    Bf = MX.sym("Bf")
    Dr = MX.sym("Dr")
    Cr = MX.sym("Cr")
    Br = MX.sym("Br")
    Imax_c = MX.sym("Imax_c")
    Caccel = MX.sym("Caccel")
    Cdecel = MX.sym("Cdecel")
    qadv = MX.sym("qadv")
    qn = MX.sym("qn")
    qalpha = MX.sym("qalpha")
    qvy = MX.sym("qvy")
    freq = MX.sym("freq")
    C_S_front = MX.sym('C_S_front')
    C_S_rear = MX.sym('C_S_rear')

    # parameters
    p = vertcat(m, C1, C2, Cr0, Cr2, Cr3, Iz, lr, lf, Bf, Cf, Df, Br, Cr, Dr, Imax_c, Caccel, Cdecel, qadv, qn, qalpha, qvy, freq, C_S_front, C_S_rear)
    
    # calculate the index
    s_mod = fmod(s, pathlength)

    # constraint on forces
    # beta = cs.atan(lr/(lr + lf) * cs.tan(delta))
    # beta = lr/(lr + lf) * (delta) # full formulation above is not relevantly different for [-0.4 - 0.4]
    a_lat = v * v * cs.tan(delta)/(lr+lf)
    a_long = der_v

    # left_bound=inner_side, right_bound=outer_side
    # distance to border
    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n 

    pos_s = MX.sym('pos_s')
    pos_n = MX.sym('pos_n')
    model.n_outer_bound = Function('outer_bound', [pos_s,pos_n], [outer_bound_s(pos_s) + pos_n])
    model.n_inner_bound = Function('inner_bound', [pos_s,pos_n], [inner_bound_s(pos_s) - pos_n])

    # Model bounds
    model.n_min = cfg_dict['track_safety_margin']  # width of the track [m]
    model.n_max = cfg_dict['track_max_width'] # width of the track [m]
    # state bounds

    model.delta_min = cfg_dict["delta_min"] # minimum steering angle [rad]
    model.delta_max = cfg_dict["delta_max"] # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = cfg_dict["ddelta_min"] # minimum change rate of stering angle [rad/s]
    model.ddelta_max = cfg_dict["ddelta_max"] # maximum change rate of steering angle [rad/s]
    model.dv_min = cfg_dict["dv_min"] # minimum velocity change rate
    model.dv_max = cfg_dict["dv_max"] # maximum velocity change rate

    # nonlinear constraint
    constraint.alat_min = cfg_dict["alat_min"] # maximum lateral force [m/s^2]
    constraint.alat_max = cfg_dict["alat_max"] # maximum lateral force [m/s^1]

    constraint.along_min = cfg_dict["along_min"] # maximum longitudinal force [m/s^2]
    constraint.along_max = cfg_dict["along_max"] # maximum longitudinal force [m/s^2]

    # Velocity constraint
    constraint.v_min = cfg_dict["v_min"] # minimum velocity [m/s]
    constraint.v_max = cfg_dict["v_max"] # maximum velocity [m/s]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n_inner_bound, n_outer_bound)

    # dynamics
    sdota = (v * cos(alpha )) / (1 - kapparef_s(s_mod) * n)
    ndot = v * sin(alpha) # (v * cos(alpha + beta))**2 * (0.5 * kapparef_s(s_mod))
    alphadot= v / (lr + lf) * cs.tan(delta) - kapparef_s(s_mod) * sdota
    # state is (s, n, alpha, v, delta) 
    f_expl = vertcat(
        sdota,
        ndot,
        alphadot,
        der_v,
        derDelta,
    )

    f_expl_func = Function('f_expl_func', [s,n,alpha,v,delta, der_v, derDelta, p], [f_expl])

    # discrete dynamics：augmented model
    # Euler forward simulation

    control_period=1/freq
    # dis_s_k_1=s+control_period*sdota
    # dis_n_k_1=n+control_period*ndot
    # dis_alpha_k_1=alpha+control_period*alphadot
    # dis_v_k_1=v+der_v
    # dis_delta_k_1=delta+derDelta
    # f_disc=vertcat(
    # dis_s_k_1,
    # dis_n_k_1,
    # dis_alpha_k_1,
    # dis_v_k_1,
    # dis_delta_k_1,
    # )
    # f_disc_func = Function('f_disc_func', [s,n,alpha,v,delta, der_v, derDelta, p], [f_disc])

    # runge kutta simulation
    f_c_runge= vertcat(
        sdota,
        ndot,
        alphadot,
    )
    x_runge = vertcat(s,n,alpha)
    f_c_runge_func = Function('f_c_runge_func', [s,n,alpha,v,delta,p], [f_c_runge])
    # arg input
    dis_v_k=v+der_v
    dis_delta_k=delta+derDelta

    k1 = control_period * f_c_runge_func(s,n,alpha,dis_v_k,dis_delta_k,p)
    x_next_runge= x_runge + 0.5 * k1
    k2 = control_period* f_c_runge_func(x_next_runge[0],x_next_runge[1],x_next_runge[2],dis_v_k,dis_delta_k,p)
    x_next_runge = x_runge + 0.5 * k2
    k3 = control_period* f_c_runge_func(x_next_runge[0],x_next_runge[1],x_next_runge[2],dis_v_k,dis_delta_k,p)
    x_next_runge = x_runge + k3
    k4 = control_period* f_c_runge_func(x_next_runge[0],x_next_runge[1],x_next_runge[2],dis_v_k,dis_delta_k,p)
    x_next_runge=x_runge + (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    f_disc=vertcat(x_next_runge,dis_v_k,dis_delta_k)

    # Definition of external cost function
    R = np.diag(cfg_dict['R'])
    qv = cfg_dict['qv']
    sdota_no_beta = (v * cos(alpha)) / (1 - kapparef_s(s_mod) * n)
    model.cost_expr_ext_cost = -qadv * sdota / freq + qn * n**2 + qalpha * alpha**2 +qv*(v-vx_ref_s(s_mod))**2+u.T @ R @ u
    model.cost_expr_ext_cost_e = -qadv * sdota_no_beta / freq +  qn * n**2 + qalpha * alpha**2+qv*(v-vx_ref_s(s_mod))

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.dyn_disc_fun = f_disc
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

