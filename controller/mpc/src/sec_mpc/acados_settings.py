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

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from sec_mpc.bicycle_model import bicycle_model
from utils.indecies_old import State
import numpy as np
from utils.indecies_old import State, Input


def acados_settings(s0, kapparef, vx_ref,d_left, d_right, cfg_dict:dict):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint, params = bicycle_model(s0, kapparef, vx_ref,d_left, d_right, cfg_dict)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    # model_ac.disc_dyn_expr = model.dyn_disc_fun
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # Set parameters dimension
    p = get_parameters(cfg_dict)
    params.p = p
    ocp.parameter_values = p


    # define constraint：[longitudinal force, lateral force, inner_bound, outer_bound]
    model_ac.con_h_expr = constraint.expr

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    nsbx = 1
    nh = constraint.expr.shape[0]
    nsh = nh
    ns = nsh + nsbx

    # discretization
    ocp.dims.N = cfg_dict['N']

    ocp.dims.ny = 0
    ocp.dims.ny_e = 0

    # use external costfunction defined in bicycle.py
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # cost coefficients of slack variables  
    # State slack variables: sl
    # Input slack varibales: su
    # Slack penalty function is 0.5*Z_l*s_l**2 + 0.5*Z_u*s_u**2 + z_l*s_l + z_u*s_u
    ocp.cost.zl = cfg_dict['zl'] * np.ones((ns,)) # linear coefficient state slack variable
    ocp.cost.zu = cfg_dict['zu'] * np.ones((ns,)) # linear coefficient input slack variable
    ocp.cost.Zl = cfg_dict['Zl'] * np.ones((ns,)) # quadratic coefficient state slack variable
    ocp.cost.Zu = cfg_dict['Zu'] * np.ones((ns,)) # quadratic coefficient input slack variable

    # Input hard constraint
    ocp.constraints.lbu = np.array([model.dv_min, model.ddelta_min])
    ocp.constraints.ubu = np.array([model.dv_max, model.ddelta_max])
    ocp.constraints.idxbu = np.array([Input.D_VELOCITY, Input.D_STEERING_ANGLE])

    # State soft constraint
    ocp.constraints.lsbx = np.zeros([nsbx])
    ocp.constraints.usbx = np.zeros([nsbx])
    ocp.constraints.idxsbx = np.array(range(nsbx))

    # Define constraints: only constrain the velocity and steering angle
    ocp.constraints.lbx = np.array(
        [
            constraint.v_min,
            model.delta_min,
        ]
    )
    ocp.constraints.ubx = np.array(
        [   
            constraint.v_max,
            model.delta_max,
        ]
    )
    # State index
    ocp.constraints.idxbx = np.array([State.VELOCITY_V, State.STEERING_ANGLE_DELTA])
    
    # Nonlinear constraint: longitudinal force; Lateral force; inner_bound; outer_bound
    ocp.constraints.lh = np.array(
        [
            0,
            0,
            0,
            0,
            -100,
            -100,
            -100,
            -100,
            model.n_min,
            model.n_min,
        ]
    )
    ocp.constraints.uh = np.array(  
        [
            100,
            100,
            100,
            100,
            0,
            0,
            0,
            0,
            model.n_max,
            model.n_max,
        ]
    )

    # Nonlinear soft constraint
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh) + np.array([0, 0, 0, 0, 0, 0, 0, 0,1, 1])
    ocp.constraints.idxsh = np.array([num for num in range(nsh)])

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    # Prediction time
    ocp.solver_options.tf = cfg_dict['N'] / cfg_dict['MPC_freq']

    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI" # SQP_RTI
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.integrator_type = "DISCRETE"
    # continuous time integrator
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3
    # ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_max_iter = 500
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.print_level = 0
    # ocp.solver_options.nlp_solver_tol_comp = 1e-1

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver, params


def get_parameters(cfg: dict) -> np.ndarray:
    """Get parameters from param_config.yaml"""
    params = np.array([
        cfg["ini_v_traget"],
        cfg["qadv"],
        cfg["qv"],
        cfg["qn"],
        cfg["qalpha"],
        cfg["v_min"],
        cfg["v_max"],
        cfg["ddelta_min"],
        cfg["ddelta_max"],
        cfg["a_min"],
        cfg["a_max"],
        cfg["alat_min"],
        cfg["alat_max"],
    ]).astype(float)
    
    return params

