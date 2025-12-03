
from typing import Tuple

import rospkg
import yaml
from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator
from pbl_config import LBMPCConfig, load_LBMPC_config_ros
from pydantic import BaseModel, ConfigDict, ValidationError


class LBMPCConfigDyn(BaseModel):
    """Learning-Based MPC dynamic reconfigure configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # Cost function settings #
    ##########################
    qadv: Tuple[float, float]
    """Limits on the advancement cost"""
    qv: Tuple[float, float]
    """Limits on the velocity cost"""
    qn: Tuple[float, float]
    """Limits on the lateral deviation cost"""
    qalpha: Tuple[float, float]
    """Limits on the heading deviation cost"""
    qjerk: Tuple[float, float]
    """Limits on the jerk cost"""
    qddelta: Tuple[float, float]
    """Limits on the steering angle rate cost"""
    alat_max: Tuple[float, float]
    """Limits on the maximum lateral acceleration"""
    a_min: Tuple[float, float]
    """Limits on the minimum acceleration (Maximum Breaking)"""
    a_max: Tuple[float, float]
    """Limits on the maximum acceleration"""
    track_safety_margin: Tuple[float, float]
    """Limits on the track safety margin"""
    terminal_multiplier: Tuple[float, float]
    """Weight multiplier for the cost of the last state"""


def create_LBMPC_dynamic_parameters(gen: ParameterGenerator):
    """Create dynamic reconfigure parameters for the single track MPC controller

    Args:
        gen (ParameterGenerator): The dynamic reconfigure parameter generator
    """
    default_car = "DEFAULT"
    # load default parameters from the yaml file
    MPCconfig: LBMPCConfig = load_LBMPC_config_ros(
        racecar_version=default_car)  # using the default_car version by default

    # load the dynamic reconfigure parameters
    relative_path = f'/config/{default_car}/single_track_mpc_dyn_params.yaml'
    config_path = rospkg.RosPack().get_path('stack_master') + relative_path
    with open(config_path, 'r', encoding='utf-8') as file:
        cfg_dict = yaml.safe_load(file)
    try:
        config = LBMPCConfigDyn(**cfg_dict)
    except ValidationError as e:
        for error in e.errors():
            if error["type"] == "missing":
                for missing_key in error["loc"]:
                    raise ValueError(f"Missing key <{missing_key}> in {config_path} file. Please add it.") from e
            elif error["type"] == "extra_forbidden":
                for extra_key in error["loc"]:
                    raise ValueError(f"Extra key <{extra_key}> in {config_path} file. Please remove it.") from e
            else:
                raise ValueError(
                    f"Error loading the {config_path} file. Please contact support (edo) with this traceback.") from e
        return None

    # check dyn parameters are available in the normal config
    for key in config.model_dump().keys():
        if key not in MPCconfig.model_dump().keys():
            raise ValueError(
                f"Key <{key}> is not available in the normal config. Please only try to dynamically reconfigure only the available parameters.")

    # create the dynamic reconfigure parameters
    for key, value in config.model_dump().items():
        gen.add(name=key,
                paramtype='double',
                level=0,
                description=f"{config.model_fields[key].description}",
                default=MPCconfig.model_dump()[key],
                min=value[0],
                max=value[1],
                )

    return gen
