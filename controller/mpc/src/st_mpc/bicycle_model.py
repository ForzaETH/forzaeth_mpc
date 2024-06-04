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


def bicycle_model(s0: list, kapparef: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "ST_model"

    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[:length -1 ] - s0[length - 1]], s0)
    kapparef = np.append(kapparef, kapparef[1:length])
    kapparef = np.append([kapparef[:length - 1] - kapparef[length - 1]], kapparef)

    d_left = np.append(d_left, d_left[1:length])
    d_left = np.append([d_left[:length -1 ] - d_left[length - 1]], d_left)
    d_right = np.append(d_right, d_right[1:length])
    d_right = np.append([d_right[:length -1 ] - d_right[length - 1]], d_right)



    N = cfg_dict['N']
    Tf = cfg_dict['Tf']


    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    #########################
    ## CasADi Model : STATE #
    #########################
    # set up state vector
    s = MX.sym("s")
    n = MX.sym("n")
    v = MX.sym("v")
    delta = MX.sym("delta")
    alpha = MX.sym("alpha")
    yaw_rate = MX.sym("yaw_rate")
    beta = MX.sym("beta")

    x = vertcat(s, n, v, delta, alpha, yaw_rate, beta)

    # inputs 
    a_long = MX.sym("a_long")
    st_vel = MX.sym("st_vel")
    u = vertcat(a_long, st_vel)

    # states in "delta" formulation
    next_v = v + Tf*a_long/N
    next_delta = delta + (Tf)*st_vel/N 

    # xdot
    s_dot = MX.sym("s_dot")
    n_dot = MX.sym("n_dot")
    v_dot = MX.sym("v_dot")
    delta_dot = MX.sym("delta_dot")
    alpha_dot = MX.sym("alpha_dot")
    yaw_rate_dot = MX.sym("yaw_rate_dot")
    beta_dot = MX.sym("beta_dot")
    
    xdot = vertcat(s_dot, n_dot, v_dot, delta_dot, alpha_dot, yaw_rate_dot, beta_dot)

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
    h_cg = MX.sym('h_cg')
    mu = MX.sym('mu')
    g = 9.81

    p = vertcat(m, C1, C2, Cr0, Cr2, Cr3, Iz, lr, lf, Bf,
                Cf, Df, Br, Cr, Dr, Imax_c, Caccel, Cdecel, qadv, qn,
                qalpha, qvy, freq, C_S_front, C_S_rear, h_cg, mu)

    s_mod = fmod(s, pathlength)


    ##########################
    ## CasADi Model : constr #
    ##########################

    # constraint on forces
    
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

    # input bounds
    model.dv_min = cfg_dict["dv_min"] # minimum velocity change rate
    model.dv_max = cfg_dict["dv_max"] # maximum velocity change rate

    # nonlinear constraint
    constraint.a_max = cfg_dict["along_max"] # maximum lateral force [m/s^2]

    # Velocity constraint
    constraint.v_min = cfg_dict["v_min"] # minimum velocity [m/s]
    constraint.v_max = cfg_dict["v_max"] # maximum velocity [m/s]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 1, 0, 0, 0, 0])

    # define constraints struct
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, beta, n_inner_bound, n_outer_bound)

    ############
    # DYNAMICS #
    ############
    sdot_expl = (next_v * cos(alpha)) / (1 - kapparef_s(s_mod) * n)
    ndot_expl = next_v * sin(alpha)
    vdot_epxl = a_long
    deltadot_expl = st_vel
    alphadot_expl = yaw_rate
    yaw_ratedot_expl = (
        mu*m/(Iz * (lr+lf))
    ) * ((
        lf*C_S_front*(g*lr-a_long*h_cg)*next_delta
        ) + (
        (lr*C_S_rear*(g*lf + a_long*h_cg) - lf*C_S_front*(g*lr - a_long*h_cg))*beta
        ) - (
        (lf**2*C_S_front*(g*lr - a_long*h_cg) + lr**2*C_S_rear*(g*lf + a_long*h_cg))*yaw_rate/next_v
        )
    )
    betadot_expl = (
        mu/(next_v * (lr+lf))
    ) * ((
        C_S_front*(g*lr-a_long*h_cg)*next_delta
        ) - (
        (C_S_rear*(g*lf + a_long*h_cg) + C_S_front*(g*lr - a_long*h_cg))*beta
        ) + (
        (lr*C_S_rear*(g*lf + a_long*h_cg) - lf*C_S_front*(g*lr - a_long*h_cg))*yaw_rate/next_v
        )
    ) - yaw_rate
    
    # state is (s, n, v, delta, alpha, yaw_rate, beta)
    f_expl = vertcat(
        sdot_expl,
        ndot_expl,
        vdot_epxl,
        deltadot_expl,
        alphadot_expl,
        yaw_ratedot_expl,
        betadot_expl
    )

    f_expl_func = Function('f_expl_func', [s, n, v, delta, alpha, yaw_rate, beta, a_long, st_vel, p], [f_expl])


    # Definition of external cost function
    R = np.diag(cfg_dict['R'])
    qv = cfg_dict['qv']
    sdota_no_beta = (v * cos(alpha)) / (1 - kapparef_s(s_mod) * n)
    model.cost_expr_ext_cost = -qadv * sdot_expl / freq - qv * v + qn * n**2 + qalpha * alpha**2 + u.T @ R @ u
    model.cost_expr_ext_cost_e = -qadv * sdota_no_beta / freq + 5 * qn * n**2 + 5 * qalpha * alpha**2

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
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
