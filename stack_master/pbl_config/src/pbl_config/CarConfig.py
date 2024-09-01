import yaml
import rospkg
from pydantic import BaseModel, ConfigDict, ValidationError

class CarConfig(BaseModel):
    """Car Parameters configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    m: float
    """Mass of the car"""
    Iz: float
    """Moment of inertia of the car"""
    lf: float
    """Distance from the center of mass to the front axle"""
    lr: float
    """Distance from the center of mass to the rear axle"""
    h_cg: float
    """Height of the center of gravity"""

def load_car_config_ros(racecar_version: str) -> CarConfig:
    """Loads the car config from the yaml file

    Args:
        racecar_version (str): a car name
    """

    relative_path = '/config/' + racecar_version + '/car_params.yaml'
    config_path = rospkg.RosPack().get_path('stack_master') + relative_path
    with open(config_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
        try:
            config = CarConfig(**cfg_dict)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "missing":
                    for missing_key in error["loc"]:
                        print(f"Missing key <{missing_key}> in {relative_path} file. Please add it.")
                elif error["type"] == "extra_forbidden":
                    for extra_key in error["loc"]:
                        print(f"Extra key <{extra_key}> in {relative_path} file. Please remove it.")
                else:
                    print(f"Error loading the {relative_path} file. Please contact support (edo) with this traceback.")
                    raise e
            return None

    return config
