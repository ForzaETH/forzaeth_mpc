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
from math import sqrt
import types
from nav_msgs.msg import Odometry
import casadi as cs
import numpy as np
from casadi import MX, Function, cos, interpolant, fmod, sin, vertcat


def bicycle_model(s0: list, kapparef: list, d_left: list, d_right: list, cfg_dict: dict):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

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



    #N = cfg_dict['N']
    #Tf = cfg_dict['Tf']


    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    outer_bound_s = interpolant("outer_bound_s", "bspline", [s0], d_left)
    inner_bound_s = interpolant("inner_bound_s", "bspline", [s0], d_right)

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    theta = MX.sym("theta")
    v_x = MX.sym("v_x")
    v_y = MX.sym("v_y")
    delta = MX.sym("delta")
    yaw_rate = MX.sym("yaw_rate")
    accel = MX.sym("accel")
    

    x = vertcat(s, n, theta, v_x, v_y, delta, yaw_rate, accel)

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
    Ef = MX.sym("Ef")
    Er = MX.sym("Er")
    rho = MX.sym("rho")
    friction_coeff = MX.sym("friction_coeff")
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

    # parameters
    p = vertcat(m, C1, C2, Cr0, Cr2, Cr3, Iz, lr, lf, Bf, Cf, Df, Br, Ef, Er, rho, friction_coeff, Cr, Dr, Imax_c, Caccel, Cdecel, qadv, qn, qalpha, qvy, freq, C_S_front, C_S_rear,h_cg,mu)

    s_mod = fmod(s, pathlength)

    # constraint on forces
    v_0 = 3
    func_v = v_x*(1/(1 + cs.exp(-v_x/v_0))) - v_x*(1/(1 + cs.exp(v_x/v_0))) + v_0*(1/(cs.cosh(v_x/v_0))) 

    alpha_f = cs.atan2((-v_y - (lf*yaw_rate)),func_v) + delta
    print(alpha_f, "THIS IS ALPHA F")
    alpha_r = cs.atan2((-v_y + (lr*yaw_rate)),func_v)  

    F_zf = m * ((-accel*h_cg)+(g*lr))/(lf+lr)
    F_zr = m * ((accel*h_cg)+(g*lf))/(lf+lr)

    F_yf = friction_coeff*Df*F_zf*sin(Cf*cs.atan(Bf*alpha_f - Ef*(Bf*alpha_f - cs.atan(Bf*alpha_f))))
    F_yr = friction_coeff*Dr*F_zr*sin(Cr*cs.atan(Br*alpha_r - Er*(Br*alpha_r - cs.atan(Br*alpha_r))))

    r_t = (cs.tan(delta)*v_x)/(lr + lf)
    m_tv = rho*(r_t - yaw_rate)

    #vel = float(sqrt((v_x)**2 + (v_y)**2))

    #beta = cs.atan(v_y/v_x)

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
    model.a_min = cfg_dict["a_min"] # minimum velocity change rate
    model.a_max = cfg_dict["a_max"] # maximum velocity change rate

    # input bounds
    model.ddelta_min = cfg_dict["ddelta_min"] # minimum change rate of stering angle [rad/s]
    model.ddelta_max = cfg_dict["ddelta_max"] # maximum change rate of steering angle [rad/s]
    model.jerk_min = cfg_dict["jerk_min"] # minimum velocity change rate
    model.jerk_max = cfg_dict["jerk_max"] # maximum velocity change rate


    # nonlinear constraint
    #constraint.alpha_f_min = cfg_dict["alpha_f_min"] # minimum FA slip[rad]
    #constraint.alpha_f_max = cfg_dict["alpha_f_max"] # maximum FA slip[rad]

    #constraint.alpha_r_min = cfg_dict["alpha_r_min"] # mimimum RA slip [rad]
    #constraint.alpha_r_max = cfg_dict["alpha_r_max"] # maximum RA slip [rad]

    # Velocity constraint
    constraint.v_x_min = cfg_dict["v_x_min"] # minimum velocity [m/s]
    constraint.v_x_max = cfg_dict["v_x_max"] # maximum velocity [m/s]

    #constraint.yaw_rate_min = cfg_dict["yaw_rate_min"] # minimum velocity [m/s]
    #constraint.yaw_rate_max = cfg_dict["yaw_rate_max"] # maximum velocity [m/s]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 1, 0, 0, 0, 0])

    # define constraints struct
    #constraint.alpha_f = Function("alpha_f", [x, u], [alpha_f])
    #constraint.alpha_r = Function("alpha_r", [x, u], [alpha_r])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(n_inner_bound, n_outer_bound)#alpha_f, alpha_r, 

    # dynamics
    sdota = ((v_x*cos(theta)) - (v_y*sin(theta))) / (1 - kapparef_s(s_mod) * n)
    ndot = v_x*sin(theta) + v_y*cos(theta) # (v * cos(alpha + beta))**2 * (0.5 * kapparef_s(s_mod))
    der_theta = yaw_rate - (kapparef_s(s_mod)*(((v_x*cos(theta)) - (v_y*sin(theta))) / (1 - kapparef_s(s_mod) * n)))
    yaw_ratedot = (1/Iz)*((-F_yr*lr)+(F_yf*lf)*cos(delta)) + m_tv
    der_v_x = accel +((1/m)*(-F_yf*sin(delta))) + (v_y*yaw_rate) 
    der_v_y = ((1/m)*(F_yr+F_yf*cos(delta))) - (v_x*yaw_rate)

    # state is (s, n, alpha, v, delta)
    f_expl = vertcat(
        sdota,
        ndot,
        der_theta,
        der_v_x,
        der_v_y,
        derDelta,
        yaw_ratedot,
        jerk
    )

    f_expl_func = Function('f_expl_func', [s, n, theta, v_x, v_y, delta, yaw_rate,accel, jerk, derDelta, p], [f_expl])


    # Definition of external cost function
    R = np.diag(cfg_dict['R'])
    qv = cfg_dict['qv']
    sdota_no_beta = ((v_x*cos(theta)) - (v_y*sin(theta))) / (1 - kapparef_s(s_mod) * n)
    model.cost_expr_ext_cost = -qadv * sdota / freq + qn * n**2 + qalpha * theta**2 + u.T @ R @ u - qv * (v_x**2 + v_y**2) 
    model.cost_expr_ext_cost_e = -qadv * sdota / freq + 5 * qn * n**2 + 5 * qalpha * theta**2

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
