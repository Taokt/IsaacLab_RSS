# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p my_tasks/dual_arm_kuka.py --robot kuka

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument("--robot", type=str, default="kuka", help="Name of the robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
# from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG, KUKA_CFG # isort:skip

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

TRASH_CAN_USD_PATH = "/home/zyf/CS_project/3D-Diffusion-Policy-LTH/third_party/IsaacLab_RSS/my_tasks/my_can.usd"

KUKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/zyf/CS_project/3D-Diffusion-Policy-LTH/third_party/IsaacLab_RSS/my_tasks/KUKA_Dual_Arm_LTH.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # joint_pos={
        #     "left_iiwa_joint_1": 0.0,
        #     "left_iiwa_joint_2": -0.569,
        #     "left_iiwa_joint_3": 0.0,
        #     "left_iiwa_joint_4": -1.810,
        #     "left_iiwa_joint_5": 0.0,
        #     "left_iiwa_joint_6": 1.037,
        #     "left_iiwa_joint_7": 0.741,
        #     "right_iiwa_joint_1": 0.0,
        #     "right_iiwa_joint_2": -0.569,
        #     "right_iiwa_joint_3": 0.0,
        #     "right_iiwa_joint_4": -1.810,
        #     "right_iiwa_joint_5": 0.0,
        #     "right_iiwa_joint_6": 1.037,
        #     "right_iiwa_joint_7": 0.741,
        # },
        joint_pos={
            "left_iiwa_joint_1": 0.0,
            "left_iiwa_joint_2": -0.569,
            "left_iiwa_joint_3": 0.0,
            "left_iiwa_joint_4": -1.810,
            "left_iiwa_joint_5": 0.0,
            "left_iiwa_joint_6": 1.037,
            "left_iiwa_joint_7": 0.741,
            "right_iiwa_joint_1": 0.0,
            "right_iiwa_joint_2": 0,
            "right_iiwa_joint_3": 0.0,
            "right_iiwa_joint_4": 0,
            "right_iiwa_joint_5": 0.0,
            "right_iiwa_joint_6": 0,
            "right_iiwa_joint_7": 0,
        },
    ),
    actuators={
        "kuka_left_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_iiwa_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "kuka_left_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_iiwa_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "kuka_left_hand": ImplicitActuatorCfg(
            joint_names_expr=["left_iiwa_gripper_.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "kuka_right_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_iiwa_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "kuka_right_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_iiwa_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "kuka_right_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_iiwa_gripper_.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

KUKA_CFG_HIGH_PD_CFG = KUKA_CFG.copy()
KUKA_CFG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_left_shoulder"].stiffness = 400.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_left_shoulder"].damping = 80.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_left_forearm"].stiffness = 400.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_left_forearm"].damping = 80.0

KUKA_CFG_HIGH_PD_CFG.actuators["kuka_right_shoulder"].stiffness = 400.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_right_shoulder"].damping = 80.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_right_forearm"].stiffness = 400.0
KUKA_CFG_HIGH_PD_CFG.actuators["kuka_right_forearm"].damping = 80.0


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    can = AssetBaseCfg(
        prim_path="/World/TrashBin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRASH_CAN_USD_PATH,
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.0, 2.0, 0.5)  # 设置垃圾桶的位置 (x, y, z)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "kuka":
        robot = KUKA_CFG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10"
        )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
    )
    goal_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
    )

    # Define goals for the arm
    # ee_goals = [
    #     [-0.5, -1.0, 0.5, 0.707, 0, 0.707, 0],
    #     [-1.0, -0.8, 0.4, 0.707, 0.707, 0.0, 0.0],
    #     [-1.0, -0.6, 0.3, 0.0, 1.0, 0.0, 0.0],
    # ]
    ee_goals = [
        [-0.5, 0.0, 0.7, 0.707, 0, 0.707, 0],
        [-0.8, -0.8, 0.6, 0.707, 0.707, 0.0, 0.0],
        [-0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(
        scene.num_envs, diff_ik_controller.action_dim, device=robot.device
    )
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[".*"], body_names=["ee_link"]
        )
    elif args_cli.robot == "kuka":
        robot_entity_cfg_left = SceneEntityCfg(
            "robot",
            joint_names=["left_iiwa_joint.*"],
            body_names=["left_iiwa_link_ee"],
        )
        robot_entity_cfg_right = SceneEntityCfg(
            "robot",
            joint_names=["right_iiwa_joint.*"],
            body_names=["right_iiwa_link_ee"],
        )
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10"
        )
    # Resolving the scene entities
    robot_entity_cfg_left.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg_left.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg_left.body_ids[0]

    print(robot_entity_cfg_left.joint_names)
    print(robot_entity_cfg_left.body_names)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            # print(joint_pos)
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg_left.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg_left.joint_ids
            ]
            ee_pose_w = robot.data.body_state_w[
                :, robot_entity_cfg_left.body_ids[0], 0:7
            ]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg_left.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )
            # print(f"set joint pos:", joint_pos_des)

        # apply actions
        robot.set_joint_position_target(
            joint_pos_des, joint_ids=robot_entity_cfg_left.joint_ids
        )
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg_left.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(
            ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7]
        )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
