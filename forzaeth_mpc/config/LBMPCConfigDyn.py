from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

import yaml
from forzaeth_mpc.config import LBMPCConfig, load_LBMPC_config_ros
from pydantic import BaseModel, ConfigDict, ValidationError

from .paths import config_root

if TYPE_CHECKING:
    from dynamic_reconfigure.parameter_generator_catkin import ParameterGenerator


class LBMPCConfigDyn(BaseModel):
    """Learning-Based MPC dynamic reconfigure configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    # Cost function settings #
    ##########################
    qadv: Tuple[float, float]
    """Limits on the advancement cost"""
    qv: Tuple[float, float]
    """Limits on the velocity cost"""
    qv_start: Tuple[float, float]
    """Limits on the velocity cost on the start mode"""
    qn: Tuple[float, float]
    """Limits on the lateral deviation cost"""
    qalpha: Tuple[float, float]
    """Limits on the heading deviation cost"""
    qjerk: Tuple[float, float]
    """Limits on the jerk cost"""
    qjerk_start: Tuple[float, float]
    """Limits on the jerk cost on the start mode"""
    qddelta: Tuple[float, float]
    """Limits on the steering angle rate cost"""
    alat_max: Tuple[float, float]
    """Limits on the maximum lateral acceleration"""
    alat_max_start: Tuple[float, float]
    """Limits on the maximum lateral acceleration on the start mode"""
    a_min: Tuple[float, float]
    """Limits on the minimum acceleration (Maximum Breaking)"""
    a_min_start: Tuple[float, float]
    """Limits on the minimum acceleration on the start mode"""
    a_max: Tuple[float, float]
    """Limits on the maximum acceleration"""
    a_max_start: Tuple[float, float]
    """Limits on the maximum acceleration on the start mode"""
    track_safety_margin: Tuple[float, float]
    """Limits on the track safety margin"""
    terminal_multiplier: Tuple[float, float]
    """Weight multiplier for the cost of the last state"""
    learning_enabled: bool = False
    """Enable or disable online learning at runtime when gp_params.data_processing_mode is 'learn'."""


def _get_dynamic_default_value(config: LBMPCConfig, key: str):
    if key == "learning_enabled":
        return config.gp_params.learning_enabled
    return getattr(config, key)


def create_LBMPC_dynamic_parameters(gen: ParameterGenerator):
    """Create dynamic reconfigure parameters for the single track MPC controller

    Args:
        gen (ParameterGenerator): The dynamic reconfigure parameter generator
    """
    from dynamic_reconfigure.parameter_generator_catkin import bool_t

    default_car = "DEFAULT"
    # load default parameters from the yaml file
    MPCconfig: LBMPCConfig = load_LBMPC_config_ros(
        racecar_version=default_car)  # using the default_car version by default

    # load the dynamic reconfigure parameters
    config_path = str(config_root() / default_car / 'lb_mpc_dyn_params.yaml')
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
        try:
            _get_dynamic_default_value(MPCconfig, key)
        except AttributeError as exc:
            raise ValueError(
                f"Key <{key}> is not available in the normal config. Please only try to dynamically reconfigure only the available parameters.")

    # create the dynamic reconfigure parameters
    for key, value in config.model_dump().items():
        if isinstance(value, bool):
            gen.add(
                name=key,
                paramtype=bool_t,
                level=0,
                description=f"{config.model_fields[key].description}",
                default=_get_dynamic_default_value(MPCconfig, key),
            )
        else:
            gen.add(name=key,
                    paramtype='double',
                    level=0,
                    description=f"{config.model_fields[key].description}",
                    default=_get_dynamic_default_value(MPCconfig, key),
                    min=value[0],
                    max=value[1],
                    )

    return gen
