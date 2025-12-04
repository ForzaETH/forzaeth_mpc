
from typing import Dict, List, Optional, Union

import rospkg
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

NestedFloats = Union[List[float], List[List[float]]]


class GPParamsConfig(BaseModel):
    """Configuration for the GP residual model"""
    model_config = ConfigDict(extra='forbid')

    use_gp: bool = True
    use_inducing_gp: bool = False
    min_points_before_using_gp: int = 100
    max_num_data_points: int = 2000

    # Mandatory: same name as in YAML
    unc_jac_G_mat_idx: List[int]
    input_feature_selection: List[int]

    # 'none' | 'record' | 'learn' | 'use_prerecorded_data'
    data_processing_mode: str = "none"

    # Only relevant when use_inducing_gp is True
    num_inducing_points: int = 10

    # Free-form hyperparameters, matched by string keys
    hyperparameters: Optional[Dict[str, NestedFloats]] = None


class LBMPCConfig(BaseModel):
    """Learning-Based MPC configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # general parameters
    N: int
    """Prediction horizon"""
    t_delay: float
    """Delay in seconds accounted for propagation before the MPC is applied"""
    steps_delay: int
    """Delays for the steering angle"""
    MPC_freq: int
    """Frequency of the MPC controller"""
    track_safety_margin: float
    """Safety margin for the track"""
    track_max_width: float
    """Maximum width of the track"""
    overtake_d: float
    """MPC Overtake path lateral tracking distance"""

    # Cost function settings #
    qjerk: float
    """jerk cost weight"""
    qddelta: float
    """steering angle rate cost weight"""
    qadv: float
    """advancement cost weight"""
    qn: float
    """lateral deviation cost weight"""
    qalpha: float
    """heading deviation cost weight"""
    qv: float
    """velocity tracking cost weight"""

    Zl: float
    """quadratic coefficient state slack variable"""
    Zu: float
    """quadratic coefficient input slack variable"""
    zl: float
    """linear coefficient state slack variable"""
    zu: float
    """linear coefficient input slack variable"""

    # Model constraints
    # state bounds
    delta_min: float
    """Minimum steering angle"""
    delta_max: float
    """Maximum steering angle"""
    v_min: float
    """Minimum velocity"""
    v_max: float
    """Maximum velocity"""
    a_min: float
    """Minimum acceleration (Maximum Breaking)"""
    a_max: float
    """Maximum acceleration"""

    # input bounds
    ddelta_min: float
    """Minimum steering angle rate"""
    ddelta_max: float
    """Maximum steering angle rate"""
    jerk_min: float
    """Minimum jerk"""
    jerk_max: float
    """Maximum jerk"""

    # nonlinear constraint
    alat_max: float
    """Maximum lateral acceleration"""

    # Cost Flags
    vy_minimization: bool
    """If true, minimize the lateral velocity"""
    adv_maximization: bool
    """If true, maximize the advancement"""

    # constraints flags
    combined_constraints: str
    """ellipse/diamond/anything else for None"""

    # model flags
    load_transfer: bool
    """If true, load transfer is considered"""
    correct_v_y_dot: bool
    """If true, the lateral velocity derivative is from the model and not approximated"""

    # Terminal multiplier
    terminal_multiplier: float

    gp_params: GPParamsConfig
    """Parameters for the GP residual model"""


def load_LBMPC_config_ros(racecar_version: str) -> Union[LBMPCConfig, None]:
    """Loads the MPC config from the yaml file

    Args:
        racecar_version (str): a car name

    Returns:
        LBMPCConfig: the MPC configuration object
    """

    relative_path = '/config/' + racecar_version + '/lb_mpc_params.yaml'
    config_path = rospkg.RosPack().get_path('stack_master') + relative_path
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg_dict = yaml.safe_load(file)
        try:
            config = LBMPCConfig(**cfg_dict)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "missing":
                    for missing_key in error["loc"]:
                        print(f"Missing key <{missing_key}> in {config_path} file. Please add it.")
                elif error["type"] == "extra_forbidden":
                    for extra_key in error["loc"]:
                        print(f"Extra key <{extra_key}> in {config_path} file. Please remove it.")
                else:
                    print(f"Error loading the {config_path} file. Please contact support (edo) with this traceback.")
                    raise e
            return None

    return config
