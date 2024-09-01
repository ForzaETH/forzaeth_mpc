
import yaml
import rospkg
from pydantic import BaseModel, ConfigDict, ValidationError

class PacejkaTireConfig(BaseModel):
    """Pacejka Tire Parameters configuration class"""
    model_config = ConfigDict(extra='forbid', use_attribute_docstrings=True)

    friction_coeff: float
    """Friction coefficient of the tire"""
    Bf: float
    """Longitudinal stiffness coefficient (B) of the FRONT tire"""
    Cf: float
    """Cornering stiffness coefficient (C) of the FRONT tire"""
    Df: float
    """Peak value of the lateral force (D) of the FRONT tire"""
    Ef: float
    """Lateral stiffness coefficient (E) of the FRONT tire"""

    Br: float
    """Longitudinal stiffness coefficient (B) of the REAR tire"""
    Cr: float
    """Cornering stiffness coefficient (C) of the REAR tire"""
    Dr: float
    """Peak value of the lateral force (D) of the REAR tire"""
    Er: float
    """Lateral stiffness coefficient (E) of the REAR tire"""

def load_pacejka_tire_config_ros(racecar_version: str) -> PacejkaTireConfig:
    """Loads the pacejka_tire config from the yaml file

    Args:
        racecar_version (str): a car name
    """

    relative_path = '/config/' + racecar_version + '/pacejka_tire_params.yaml'
    config_path = rospkg.RosPack().get_path('stack_master') + relative_path
    with open(config_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
        try:
            config = PacejkaTireConfig(**cfg_dict)
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
