#!/usr/bin/env python3
import rospy
import rospkg
import yaml, os
from dynamic_reconfigure.server import Server
from controller.cfg import dyn_mpc_params_tunerConfig
from pbl_config import STMPCConfig, load_STMPC_config_ros, STMPCConfigDyn

class CallbackHandler:
    def __init__(self) -> None:
        self.dyn_config_keys = STMPCConfigDyn.model_fields.keys()

    def callback(self, config: dyn_mpc_params_tunerConfig, level):
        rospy.loginfo("[STMPC dynamic parameter server] MPC Parameters Updated")
        print(config)

        for k in self.dyn_config_keys:
            # Ensuring rounding to the second decimal
            setattr(config, k, round(getattr(config, k), 2))

        return config

def get_default_dict():
    # get full config from yaml
    racecar_version = rospy.get_param('/racecar_version') # NUCX or SIM
    #TODO: automatically recongize mpc version with rosparam
    stmpc_config: STMPCConfig = load_STMPC_config_ros(racecar_version)

    # only iterate in possible keys of the dynamic reconfigure
    dyn_config_keys = STMPCConfigDyn.model_fields.keys()

    default_config = {key: getattr(stmpc_config, key) for key in dyn_config_keys}

    return default_config


if __name__ == "__main__":
    rospy.init_node("mpc_dyn_rec_tuner_server", anonymous=False)
    cbh = CallbackHandler()
    server = Server(dyn_mpc_params_tunerConfig, cbh.callback)

    default_dict = get_default_dict()

    server.update_configuration(default_dict)
    rospy.spin()
