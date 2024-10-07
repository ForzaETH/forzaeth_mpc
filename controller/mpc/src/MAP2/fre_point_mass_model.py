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

import math
import types
# author: Daniel Kloeser
from math import sqrt

import casadi as cs
import numpy as np
from casadi import MX, Function, cos, interpolant, mod, sin, vertcat
from nav_msgs.msg import Odometry


def fre_point_mass_model(s0: list, kapparef: list, vx_ref: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "fre_point_mass_model"
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end

    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    s0 = np.append([s0[:length - 1] - s0[length - 1]], s0)
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
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    theta = MX.sym("theta")
    v = MX.sym("v")
    x = vertcat(s, n, theta,v)

    # controls
    accel_x = MX.sym("accel_x")
    yawrate=MX.sym("yawrate")
    u = vertcat(accel_x,yawrate)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    thetadot = MX.sym("thetadot")
    v_dot = MX.sym("v_dot")
    xdot = vertcat(sdot, ndot, thetadot,v_dot)

    # algebraic variables
    z = vertcat([])

    #online parameters
    weight_qn = MX.sym('weight_qn')
    weight_theta = MX.sym('weight_theta')
    weight_accel_x = MX.sym('weight_accel_x')
    weight_beta=MX.sym('weight_accel_y')
    weight_yaw_rate = MX.sym('weight_yaw_rate')
    track_speed = MX.sym("track_speed")
    weight_vx=MX.sym("weight_vx")
    GP_beta_comp=MX.sym("GP_beta_comp")
    p = vertcat(weight_qn,weight_theta,weight_accel_x,weight_beta,weight_vx,weight_yaw_rate,track_speed,GP_beta_comp)

    s_mod = mod(s, pathlength)


    #Bounds
    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n

    pos_s = MX.sym('pos_s')
    pos_n = MX.sym('pos_n')
    model.n_outer_bound = Function('outer_bound', [pos_s,pos_n], [outer_bound_s(pos_s) + pos_n])
    model.n_inner_bound = Function('inner_bound', [pos_s,pos_n], [inner_bound_s(pos_s) - pos_n])

    # boudnary bounds
    model.n_min = cfg_dict['track_safety_margin']   # width of the track [m]
    model.n_max = cfg_dict['track_max_width'] # width of the track [m]


    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0])

    # define constraints struct
    # also need soft constraint
    constraint.pathlength = pathlength
    constraint.expr_0 = vertcat(n_inner_bound, n_outer_bound)
    constraint.expr = vertcat(n_inner_bound, n_outer_bound)
    constraint.expr_e = vertcat(n_inner_bound, n_outer_bound)
    # dynamics
    #simple approximation of beta plus GP correction
    beta=cs.asin(cfg_dict['lr']*yawrate/(v+0.5))+GP_beta_comp

    #calculate the beta with considering tire model
    # Fzr=cfg_dict['m']*cfg_dict['g']*cfg_dict['lf']/(cfg_dict['lf']+cfg_dict['lr'])
    # k_rear_tire=cfg_dict['mu']*Fzr*cfg_dict['C_Sr']
    # # k_rear_tire=cfg_dict['mu']*Fzr*cfg_dict['Br']*cfg_dict['Cr']*cfg_dict['Dr']
    # k_1=cfg_dict['m']*cfg_dict['lf']/((cfg_dict['lf']+cfg_dict['lr']))/k_rear_tire
    # k_2=cfg_dict['lr']
    # beta=k_1*v**2*curvature+k_2*curvature

    sdota = v*cos(theta+beta) / (1 - kapparef_s(s_mod) * n)
    ndot =  v*sin(theta+beta)
    der_theta = yawrate- (kapparef_s(s_mod)*(( v*cos(theta+beta)) / (1 - kapparef_s(s_mod) * n)))


    # state is (s, n, alpha, v)
    f_expl = vertcat(
        sdota,
        ndot,
        der_theta,
        accel_x,
    )

    #cost
    terminal_multiplier = 10
    model.cost_expr_ext_cost_0 = weight_qn*(n)**2 + weight_theta*theta**2+weight_vx*(v-track_speed)**2+weight_yaw_rate*(yawrate)**2+ weight_accel_x*accel_x**2+weight_beta*beta**2
    model.cost_expr_ext_cost = weight_qn*(n)**2 + weight_theta*theta**2+weight_vx*(v-track_speed)**2 +weight_yaw_rate*(yawrate)**2+ weight_accel_x*accel_x**2+weight_beta*beta**2
    model.cost_expr_ext_cost_e = weight_qn*(n)**2 + weight_theta*theta**2+weight_vx*(v-track_speed)**2
    model.cost_expr_ext_cost_e=terminal_multiplier*model.cost_expr_ext_cost_e
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
    model.outer_bound_s = outer_bound_s
    model.inner_bound_s = inner_bound_s
    return model, constraint, params


