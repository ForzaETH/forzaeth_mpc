#!/usr/bin/env python3
import rospy
import rospkg
import yaml, os
from dynamic_reconfigure.server import Server
from controller.cfg import dyn_mpc_params_tunerConfig

def get_racecar_version():
    racecar_version = rospy.get_param('/racecar_version') # NUCX or SIM
    print('- RACECAR VERSION: ' + racecar_version) # TODO remove once tested it works
    return racecar_version

def get_yaml_values():
    # Get corresponding yaml file path
    yaml_path = rospkg.RosPack().get_path("controller")
    # Get and return data
    with open(yaml_path + "/mpc/" + "/config" + "/mpc_params.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data

def callback(config, level):
    rospy.loginfo("L1 Parameter Updated")
    # Ensuring nice rounding by 0.05
    config.qadv = round(config.qadv * 200) / 200 # round to 0.005
    config.qv = round(config.qv * 200) / 200 # round to 0.005
    config.qn = round(config.qn * 200) / 200 # round to 0.005
    config.qalpha = round(config.qalpha * 200) / 200 # round to 0.005

    # constraints
    config.ddelta_min = round(config.ddelta_min * 200) / 200 # round to 0.005
    config.ddelta_max = round(config.ddelta_max * 200) / 200 # round to 0.005
    config.dv_min = round(config.dv_min * 200) / 200 # round to 0.005
    config.dv_max = round(config.dv_max * 200) / 200 # round to 0.005
    config.v_min = round(config.v_min * 200) / 200 # round to 0.005
    config.v_max = round(config.v_max * 200) / 200 # round to 0.005
    config.alat_min = round(config.alat_min * 200) / 200 # round to 0.005
    config.alat_max = round(config.alat_max * 200) / 200 # round to 0.005


    
    return config

if __name__ == "__main__":
    rospy.init_node("dyn_l1_params_tuner_server", anonymous=False)
    server = Server(dyn_mpc_params_tunerConfig, callback)

    yaml_data = get_yaml_values()
 
    # Set mpc parameter values to the values of the corresponding .yaml file
    # float() is need as we want to ensure its type to access it in L1_controller with "params.doubles[idx].value"
    default_config = {
        ## mpc Controller Parameters
        "qadv": float(yaml_data["qadv"]),
        "qv": float(yaml_data["qv"]),
        "qn": float(yaml_data["qn"]),
        "qalpha": float(yaml_data["qalpha"]),
        ## mpc constraints
        "ddelta_min": float(yaml_data["ddelta_min"]),
        "ddelta_max": float(yaml_data["ddelta_max"]),
        "dv_min": float(yaml_data["dv_min"]),
        "dv_max": float(yaml_data["dv_max"]),
        "v_min": float(yaml_data["v_min"]),
        "v_max": float(yaml_data["v_max"]),
        "alat_min": float(yaml_data["alat_min"]),
        "alat_max": float(yaml_data["alat_max"])
    }

    server.update_configuration(default_config)
    rospy.spin()


