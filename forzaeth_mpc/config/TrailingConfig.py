import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .paths import config_root, strict_safe_load


class TrailingConfig(BaseModel):
    """Car Parameters configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    allow_accel_trailing: bool
    """allow_accel_trailing"""
    trailing_mode: bool
    """trailing_mode"""
    prioritize_dyn: bool
    """prioritize_dyn"""
    trailing_gap: float
    """trailing_gap"""
    trailing_p_gain: float
    """trailing_p_gain in the PID controller"""
    trailing_i_gain: float
    """trailing_i_gain in the PID controller"""
    trailing_d_gain: float
    """trailing_d_gain in the PID controller"""
    blind_trailing_speed: float
    """blind_trailing_speed"""


def load_trailing_config_ros(racecar_version: str) -> TrailingConfig:
    """Loads the car config from the yaml file

    Args:
        racecar_version (str): a car name
    """

    config_path = str(config_root() / racecar_version / 'trailing_params.yaml')
    with open(config_path, 'r') as file:
        cfg_dict = strict_safe_load(file)
        try:
            config = TrailingConfig(**cfg_dict)
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
