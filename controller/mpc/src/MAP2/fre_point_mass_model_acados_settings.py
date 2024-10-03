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
from . fre_point_mass_model import fre_point_mass_model
import numpy as np


def acados_settings(s0, kapparef, vx_ref, d_left, d_right, cfg_dict:dict):
    # create render arguments
    ocp = AcadosOcp()

    # export model
    model, constraint, params = fre_point_mass_model(s0, kapparef,vx_ref ,d_left, d_right, cfg_dict)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    # define constraint：[inner_bound, outer_bound]
    model_ac.con_h_expr_0 = constraint.expr_0
    model_ac.con_h_expr = constraint.expr
    model_ac.con_h_expr_e = constraint.expr_e
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac


    # Set parameters dimension
    # initinal parameters
    p = get_parameters(cfg_dict)
    params.p = p
    ocp.parameter_values = p

    # define the number of soft constraints
    nsbx = 0  #state soft constraint
    nsbu = 1
    nsh = 2
    ns = nsh + nsbx + nsbu

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]


    # discretization
    ocp.dims.N = cfg_dict['N']

    ocp.dims.ny = 0
    ocp.dims.ny_e = 0

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost_0
    ocp.model.cost_expr_ext_cost = model.cost_expr_ext_cost
    ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e

    # input constraints
    input_constraint_min=np.array([cfg_dict['ax_min'], -3])
    input_constraint_max=np.array([cfg_dict['ax_max'], 3])
    input_constraint_index=np.array([0,1])
    ocp.constraints.lbu_0 = input_constraint_min
    ocp.constraints.ubu_0 = input_constraint_max
    ocp.constraints.idxbu_0 = input_constraint_index

    ocp.constraints.lbu = input_constraint_min
    ocp.constraints.ubu = input_constraint_max
    ocp.constraints.idxbu = input_constraint_index

    ocp.constraints.lbu_e = input_constraint_min
    ocp.constraints.ubu_e = input_constraint_max
    ocp.constraints.idxbu_e = input_constraint_index

    ocp.constraints.lbx = np.array(
        [
            0.1,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            10,
        ]
    )
    ocp.constraints.idxbx = np.array([3])

    #bound constraints
    Nonlinear_constraint_min = np.array(
        [
            model.n_min,
            model.n_min,
        ]
    )
    Nonlinear_constraint_max = np.array(
        [

            model.n_max,
            model.n_max,
        ]
    )
    #bound constraints
    Nonlinear_constraint_min_1 = np.array(
        [
            model.n_min,
            model.n_min,
        ]
    )
    Nonlinear_constraint_max_1 = np.array(
        [

            model.n_max,
            model.n_max,
        ]
    )

    Nonlinear_soft_min_value=np.zeros(2)
    Nonlinear_soft_max_value=np.zeros(2) + np.array([1, 1])
    Nonlinear_soft_min_value_1=np.zeros(2)
    Nonlinear_soft_max_value_1=np.zeros(2) + np.array([1, 1])
    Nonlinear_constraint_index_1 = np.array([0,1])
    Nonlinear_constraint_index = np.array([0,1])
    # # at stage 0
    ocp.constraints.lh_0 = Nonlinear_constraint_min_1
    ocp.constraints.uh_0 = Nonlinear_constraint_max_1
    # soft constraint
    ocp.constraints.lsh_0 = Nonlinear_soft_min_value_1
    ocp.constraints.ush_0 = Nonlinear_soft_max_value_1
    ocp.constraints.idxsh_0 = Nonlinear_constraint_index_1

    # from stage 1 to N-1
    ocp.constraints.lh = Nonlinear_constraint_min_1
    ocp.constraints.uh = Nonlinear_constraint_max_1
    # soft constraint
    ocp.constraints.lsh = Nonlinear_soft_min_value_1
    ocp.constraints.ush = Nonlinear_soft_max_value_1
    ocp.constraints.idxsh = Nonlinear_constraint_index_1

    # # at stage N
    ocp.constraints.lh_e = Nonlinear_constraint_min
    ocp.constraints.uh_e = Nonlinear_constraint_max
    # soft constraint
    ocp.constraints.lsh_e = Nonlinear_soft_min_value
    ocp.constraints.ush_e = Nonlinear_soft_max_value
    ocp.constraints.idxsh_e = Nonlinear_constraint_index


    # #input soft weights
    # # soft constraints for u
    inf=1e3
    ocp.constraints.lsbu = np.array([inf])
    ocp.constraints.usbu = np.array([inf])
    ocp.constraints.idxsbu = np.array([0])


    # slack variables included state, input and nonlinear constraints
    # minmum  slack variables: sl
    # maximum slack varibales: su

    # # Slack penalty function is Z_l*s_l**2 + Z_u*s_u**2
    # # at stage 0
    ns=ns
    ocp.cost.Zl_0 = cfg_dict['Zl'] * np.ones((ns,))
    ocp.cost.Zu_0 = cfg_dict['Zu'] * np.ones((ns,))
    ocp.cost.zl_0 = cfg_dict['Zl'] * np.ones((ns,))  # must have in the cost function, but not used (quadratic cost is enough to handle the constraints)
    ocp.cost.zu_0 = cfg_dict['Zu'] * np.ones((ns,))
    # # from 1 to N-1
    ocp.cost.Zl = cfg_dict['Zl'] * np.ones((ns,))
    ocp.cost.Zu = cfg_dict['Zu'] * np.ones((ns,))
    ocp.cost.zl = cfg_dict['Zl'] * np.ones((ns,))
    ocp.cost.zu = cfg_dict['Zu'] * np.ones((ns,))
    # # at stage N
    ocp.cost.Zl_e = cfg_dict['Zl'] * np.ones((nsh,))
    ocp.cost.Zu_e = cfg_dict['Zu'] * np.ones((nsh,))
    ocp.cost.zl_e = cfg_dict['Zl'] * np.ones((nsh,))
    ocp.cost.zu_e = cfg_dict['Zu'] * np.ones((nsh,))

    # set intial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = cfg_dict['N']/cfg_dict['MPC_freq']

    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI" # SQP_RTI
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.print_level = 0

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    return constraint, model, acados_solver, params

def get_parameters(cfg: dict) -> np.ndarray:
    """Get parameters from param_config.yaml"""
    params = np.array([
        cfg["weight_qn"],
        cfg["weight_theta"],
        cfg["weight_accel_x"],
        cfg["weight_beta"],
        cfg["weight_vx"],
        cfg["weight_yaw_rate"],
        cfg["Init_track_speed"],
        0,
    ]).astype(float)
    return params



