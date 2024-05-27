
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
    v = MX.sym("v")
    delta = MX.sym("delta")

    x = vertcat(s, n, alpha, v, delta)

    # controls
    der_v = MX.sym("der_v")
    derDelta = MX.sym("derDelta")
    u = vertcat(der_v, derDelta)

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
    R = np.diag(cfg_dict['R'])
    lr = cfg_dict['lr']
    lf = cfg_dict['lf']
    freq = cfg_dict['MPC_freq']

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

    # kinematic model
    sdota = (v * cos(alpha )) / (1 - kapparef_s(s_mod) * n)
    ndot = v * sin(alpha) # (v * cos(alpha + beta))**2 * (0.5 * kapparef_s(s_mod))
    alphadot= v / (lr + lf) * cs.tan(delta) - kapparef_s(s_mod) * sdota
    # continuous dynamics
    f_expl = vertcat(
        sdota,
        ndot,
        alphadot,
        der_v,
        derDelta,
    )
    f_expl_func = Function('f_expl_func', [s,n,alpha,v,delta, der_v, derDelta, p], [f_expl])

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0])

    # Definition of external cost function
    model.cost_expr_ext_cost =  -0*weight_adv*sdota**2/freq+weight_qn * n**2 + weight_qalpha * alpha**2 +weight_qv*(v-V_target)**2+u.T @ R @ u
    model.cost_expr_ext_cost_e = -0*weight_adv*sdota**2/freq+weight_qn * n**2 + weight_qalpha * alpha**2+weight_qv*(v-V_target)**2

    # constraint on forces
    a_lat = v * v * cs.tan(delta)/(lr+lf)

    # constraint on lateral erros
    # distance to border
    n_outer_bound = outer_bound_s(s_mod) + n
    n_inner_bound = inner_bound_s(s_mod) - n 

    pos_s = MX.sym('pos_s')
    pos_n = MX.sym('pos_n')
    model.n_outer_bound = Function('outer_bound', [pos_s,pos_n], [outer_bound_s(pos_s) + pos_n])
    model.n_inner_bound = Function('inner_bound', [pos_s,pos_n], [inner_bound_s(pos_s) - pos_n])

    # nonlinear constraints
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    # construct online parameter turning constraints
    # e.g. 0<v-cs_vx_min<inf, -inf<v-cs_vx_max<0 equivalent to v>cs_vx_min, v<cs_vx_max 
    diamond_acceleration_constraint=0*(a_lat/cs_a_lat_max)**2+0*(der_v/cs_delta_v_max)**2
    dynamic_turn_constraint_min=vertcat(v-cs_vx_min,der_v+10,derDelta-cs_delta_steer_min,a_lat-cs_a_lat_min)
    dynamic_turn_constraint_max=vertcat(v-cs_vx_max,der_v-10,derDelta-cs_delta_steer_max,a_lat-cs_a_lat_max)
    constraint.expr = vertcat(dynamic_turn_constraint_min, dynamic_turn_constraint_max, n_inner_bound, n_outer_bound,diamond_acceleration_constraint)

    ### read the parameters range from the yaml file
    # Model bounds
    model.n_min = cfg_dict['track_safety_margin']  # width of the track [m]
    model.n_max = cfg_dict['track_max_width'] # width of the track [m]
    # state bounds

    model.delta_min = cfg_dict["delta_min"] # minimum steering angle [rad]
    model.delta_max = cfg_dict["delta_max"] # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = cfg_dict["ddelta_min"] # minimum change rate of stering angle [rad/s]
    model.ddelta_max = cfg_dict["ddelta_max"] # maximum change rate of steering angle [rad/s]
    model.dv_min = cfg_dict["a_min"] # minimum velocity change rate
    model.dv_max = cfg_dict["a_max"] # maximum velocity change rate

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

