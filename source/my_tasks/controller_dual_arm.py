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
     ./isaaclab.sh -p my_tasks/controller_dual_arm.py --robot kuka

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument(
    "--robot", type=str, default="franka_panda", help="Name of the robot."
)
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
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

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
# TRASH_CAN_USD_PATH = "/home/ji0341li/CS_Project/3D-Diffusion-Policy-LTH/third_party/IsaacLab_RSS/source/my_tasks/can.usd"
TRASH_CAN_USD_PATH = "/workspace/isaaclab/source/my_tasks/can.usd"
KUKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/my_tasks/iiwa7.usd",
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
        pos=(0.0, -0.1, 0.05),  # 左侧机械臂基座位置 (x, y, z)
        rot=(0.924, 0.383, 0.0, 0.0),  # 左侧机械臂基座方向 (四元数: w, x, y, z)
        joint_pos={
            "A1": 0.0,
            "A2": -0.569,
            "A3": 0.0,
            "A4": -2.810,
            "A5": 0.0,
            "A6": 3.037,
            "A7": 0.741,
            "hande_joint_finger": 0.0,
            "robotiq_hande_base_to_hande_right_finger": 0.0,
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
KUKA_CFG_2 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/my_tasks/iiwa7.usd",
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
        pos=(0.0, 0.1, 0.05),  # 左侧机械臂基座位置 (x, y, z)
        rot=(0.924, -0.383, 0.0, 0.0),  # 左侧机械臂基座方向 (四元数: w, x, y, z)
        joint_pos={
            "A1": 0.0,
            "A2": -0.569,
            "A3": 0.0,
            "A4": -2.810,
            "A5": 0.0,
            "A6": 3.037,
            "A7": 0.741,
            "hande_joint_finger": 0.0,
            "robotiq_hande_base_to_hande_right_finger": 0.0,
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

    # Set Cube as object
    object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.5, 0, 0.055], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            usd_path=TRASH_CAN_USD_PATH,
            scale=(0.8, 0.8, 0.8),
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "kuka":
        robot = KUKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_2 = KUKA_CFG_2.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10"
        )


def initialize_controller(sim, scene, robot_type):
    """Initialize the controller and configuration for the robot."""
    # Initialize the controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Define robot-specific parameters
    if robot_type == "franka_panda":
        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
    elif robot_type == "ur10":
        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[".*"], body_names=["ee_link"]
        )
    elif robot_type == "kuka":
        robot_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[".*"], body_names=["hande_robotiq_hande_base_link"]
        )
    else:
        raise ValueError(
            f"Robot {robot_type} is not supported. Valid: franka_panda, ur10, kuka"
        )

    robot_entity_cfg.resolve(scene)

    # Determine the frame index for the end-effector
    if scene["robot"].is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    return diff_ik_controller, robot_entity_cfg, ee_jacobi_idx


def reset_simulation(
    robot, robot_entity_cfg, diff_ik_controller, ik_commands, ee_goals, current_goal_idx
):
    """Reset the simulation, including robot joint states and IK commands."""
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    ik_commands[:] = ee_goals[current_goal_idx]
    print(ee_goals[current_goal_idx])
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    # Update the goal index
    current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
    return current_goal_idx


def compute_joint_commands(robot, robot_entity_cfg, diff_ik_controller, ee_jacobi_idx):
    """Compute joint commands using the differential IK controller."""
    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
    ]
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

    # Compute the end-effector pose in the base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    # Compute joint commands
    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    return joint_pos_des


def transform_pose(robot, ee_goals):
    """
    Convert local end-effector goals to world coordinates based on the robot's base pose.

    Args:
        robot: The robot object containing the root state and other configurations.
        ee_goals (torch.Tensor): Local end-effector goals in the format [x, y, z, qx, qy, qz, qw].

    Returns:
        ee_goals_world (torch.Tensor): Transformed goals in the world frame.
    """
    # Extract the robot's base pose in the world frame
    robot_base_pose = robot.data.root_state_w[:, 0:7]  # [x, y, z, qw, qx, qy, qz]
    base_pos = robot_base_pose[:, 0:3]  # Base position [x, y, z]
    base_quat = robot_base_pose[:, 3:7]  # Base quaternion [qw, qx, qy, qz]

    # Prepare tensor for world goals
    ee_goals_world = torch.zeros_like(ee_goals, device=ee_goals.device)

    for i in range(ee_goals.shape[0]):
        # Extract local position and orientation of the goal
        local_pos = ee_goals[i, 0:3]
        local_quat = ee_goals[i, 3:7]

        # Compute world position
        world_pos = base_pos[0] + rotate_vector_by_quaternion(local_pos, base_quat[0])

        # Compute world quaternion: q_world = q_base * q_local
        base_quat_np = (
            base_quat[0].cpu().numpy()
        )  # Convert to NumPy for quaternion multiplication
        local_quat_np = local_quat.cpu().numpy()
        world_quat_np = quaternion_multiply(base_quat_np, local_quat_np)

        # Store transformed pose
        ee_goals_world[i, 0:3] = torch.tensor(world_pos, device=ee_goals.device)
        ee_goals_world[i, 3:7] = torch.tensor(world_quat_np, device=ee_goals.device)

    return ee_goals_world


def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotate a 3D vector by a quaternion.

    Args:
        vector (torch.Tensor): A 3D vector [x, y, z].
        quaternion (torch.Tensor): A quaternion [qw, qx, qy, qz].

    Returns:
        rotated_vector (torch.Tensor): Rotated vector in the world frame.
    """
    qw, qx, qy, qz = quaternion
    x, y, z = vector

    # Quaternion rotation formula
    t2 = qw * qx
    t3 = qw * qy
    t4 = qw * qz
    t5 = -qx * qx
    t6 = qx * qy
    t7 = qx * qz
    t8 = -qy * qy
    t9 = qy * qz
    t10 = -qz * qz

    rx = 2.0 * ((t8 + t10) * x + (t6 - t4) * y + (t3 + t7) * z) + x
    ry = 2.0 * ((t4 + t6) * x + (t5 + t10) * y + (t9 - t2) * z) + y
    rz = 2.0 * ((t7 - t3) * x + (t2 + t9) * y + (t5 + t8) * z) + z

    return torch.tensor([rx, ry, rz], device=vector.device)


def quaternion_multiply(q1, q2):
    """
    Perform quaternion multiplication q1 * q2.

    Args:
        q1 (numpy.ndarray): Quaternion [qw, qx, qy, qz].
        q2 (numpy.ndarray): Quaternion [qw, qx, qy, qz].

    Returns:
        numpy.ndarray: Result of quaternion multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def run_simulator(sim, scene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]

    # Initialize the controller
    diff_ik_controller, robot_entity_cfg, ee_jacobi_idx = initialize_controller(
        sim, scene, args_cli.robot
    )

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
    )
    goal_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
    )

    # Define goals for the arm
    ee_goals = torch.tensor(
        [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ],
        device=sim.device,
    )

    ee_globel_goals = transform_pose(robot, ee_goals)

    current_goal_idx = 0
    ik_commands = torch.zeros(
        scene.num_envs, diff_ik_controller.action_dim, device=robot.device
    )

    # Simulation parameters
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
            ]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # print("obtain quantities from simulation", root_pose_w, joint_pos)
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            print("compute frame in root frame", ee_pos_b, ee_quat_b)
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )
        # if count % 300 == 0:
        #     current_goal_idx = reset_simulation(
        #         robot,
        #         robot_entity_cfg,
        #         diff_ik_controller,
        #         ik_commands,
        #         ee_globel_goals,
        #         current_goal_idx,
        #     )
        # else:
        #     joint_pos_des = compute_joint_commands(
        #         robot, robot_entity_cfg, diff_ik_controller, ee_jacobi_idx
        #     )
        #     robot.set_joint_position_target(
        #         joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
        #     )

        # apply actions
        robot.set_joint_position_target(
            joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
        )

        # Perform simulation step
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # Visualization
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
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
