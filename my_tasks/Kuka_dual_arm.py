# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
KUKA_DUAL_ARM = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path = "kuka_dual_arm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_iiwa_joint_1": 0.0,
            "left_iiwa_joint_2": 0.0,
            "left_iiwa_joint_3": 0.0,
            "left_iiwa_joint_4": 0.0,
            "left_iiwa_joint_5": 0.0,
            "left_iiwa_joint_6": 0.0,
            "left_iiwa_joint_7": 0.0,
            "left_iiwa_gripper_finger_joint": 0.0,
            "right_iiwa_joint_1": 0.0,
            "right_iiwa_joint_2": 0.0,
            "right_iiwa_joint_3": 0.0,
            "right_iiwa_joint_4": 0.0,
            "right_iiwa_joint_5": 0.0,
            "right_iiwa_joint_6": 0.0,
            "right_iiwa_joint_7": 0.0,
            "right_iiwa_gripper_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=100.0,
            stiffness=1000000.0,
            damping=40.0,
        ),
    },
)

