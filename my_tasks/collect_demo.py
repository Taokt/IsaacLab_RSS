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
    ./isaaclab.sh -p my_tasks/dual_arm_kuka_control.py --robot kuka

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
from omni.isaac.lab.app import AppLauncher
from scipy.spatial.transform import Rotation
import sys
from pathlib import Path

# 添加 dual_arm_se3_keyboard 的父目录到 Python 搜索路径
keyboard_module_path = Path(
    "/home/zyf/CS_project/IsaacLabExtensionTemplate/exts/KUKA_Project/KUKA_Project/devices/keyboard"
)
sys.path.append(str(keyboard_module_path))

from dual_arm_se3_keyboard import DualArmSe3Keyboard

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
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
import threading
import time
import random
import os
import numpy as np

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

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)


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
        usd_path="/home/zyf/CS_project/3D-Diffusion-Policy-LTH/third_party/IsaacLab_RSS/my_tasks/kuka_dual_arm.usd",
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
        joint_pos={
            "left_iiwa_joint_1": 0.0,
            "left_iiwa_joint_2": -0.569,
            "left_iiwa_joint_3": 0.0,
            "left_iiwa_joint_4": -1.810,
            "left_iiwa_joint_5": 0.0,
            "left_iiwa_joint_6": 1.037,
            "left_iiwa_joint_7": 0.741,
            "right_iiwa_joint_1": 0.0,
            "right_iiwa_joint_2": -0.569,
            "right_iiwa_joint_3": 0.0,
            "right_iiwa_joint_4": -1.810,
            "right_iiwa_joint_5": 0.0,
            "right_iiwa_joint_6": 1.037,
            "right_iiwa_joint_7": 0.741,
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
    },  # type: ignore
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
        prim_path="{ENV_REGEX_NS}/Can",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[1.5, 0, 0.055], rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRASH_CAN_USD_PATH,
            scale=(0.8, 0.8, 0.8),
        ),
    )

    prim_utils.create_prim("/World/Origin_00", "Xform")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    camera = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5),
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


def multiply_quaternions(q1, q2):
    """
    Compute the product of two quaternions.

    Args:
        q1: First quaternion (array-like of shape [4]), in the form [w, x, y, z].
        q2: Second quaternion (array-like of shape [4]), in the form [w, x, y, z].

    Returns:
        A numpy array representing the resulting quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def control_single_arm(
    robot,
    robot_entity_cfg,
    diff_ik_controller,
    ee_jacobi_idx,
    delta_pose=None,  # 从键盘获取的运动增量
):
    # 获取当前末端执行器的实际位姿
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
    ]

    # 计算当前末端执行器的位姿（在根坐标系下）
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )

    # 如果有键盘输入，更新目标位姿
    if delta_pose is not None:
        ee_pos_b += delta_pose[:3]  # 增加位置偏移
        ee_quat_delta = Rotation.from_rotvec(delta_pose[3:6]).as_quat()
        ee_quat_b = multiply_quaternions(ee_quat_b, ee_quat_delta)  # 增加旋转偏移

    # 计算新的关节位置命令
    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    return joint_pos_des


def farthest_point_sample(point_cloud, npoints):
    """
    Args:
        point_cloud: 输入的点云数据，形状为 (N, 3)，其中 N 是点数，3 是坐标维度 (x, y, z)。
        npoints: 需要采样的点数。

    Returns:
        sampled_points: 采样后的点云，形状为 (npoints, 3)。
    """
    # 获取点云的数量
    N, D = point_cloud.shape
    assert npoints <= N, "采样点数不能超过点云总数！"

    # 初始化
    device = point_cloud.device  # 获取点云所在设备（CPU/GPU）
    centroids = torch.zeros(npoints, dtype=torch.long, device=device)  # 存储采样点索引
    distance = torch.ones(N, device=device) * 1e10  # 初始化每个点的最小距离
    farthest = torch.randint(0, N, (1,), device=device)  # 随机选取第一个采样点索引

    # 初始化采样点的坐标
    farthest_points = torch.zeros((npoints, D), device=device)  # 存储所有采样点的坐标

    for i in range(npoints):
        centroids[i] = farthest  # 保存当前采样点索引
        farthest_points[i] = point_cloud[farthest]  # 保存当前采样点的坐标

        # 计算所有点到当前采样点的距离
        dist = ((point_cloud - farthest_points[i]) ** 2).sum(dim=-1)

        # 使用 torch.minimum 并行更新最小距离
        distance = torch.minimum(distance, dist)

        # 找到距离最远的点索引（并行计算 argmax）
        farthest = torch.argmax(distance)

    # 根据采样点索引提取点云
    sampled_points = point_cloud[centroids]
    return sampled_points


def create_trajectory(set_points, npoints):
    """
    Generate a trajectory given a list of set points and the total number of points.

    Args:
        set_points (list of list): List of set points, where each set point is [x, y, z, qw, qx, qy, qz].
        npoints (int): Total number of points in the trajectory.

    Returns:
        trajectory (list of list): A list of points representing the trajectory.
    """
    # Ensure we have at least two set points
    if len(set_points) < 2:
        raise ValueError(
            "At least two set points are required to generate a trajectory."
        )

    # Total number of segments
    num_segments = len(set_points) - 1

    # Points per segment
    points_per_segment = npoints // num_segments

    # Initialize trajectory
    trajectory = []

    # Generate points for each segment
    for i in range(num_segments):
        start = np.array(set_points[i])
        end = np.array(set_points[i + 1])

        # Linearly interpolate between start and end
        for t in np.linspace(0, 1, points_per_segment, endpoint=False):
            interpolated = (1 - t) * start + t * end
            trajectory.append(interpolated.tolist())

    # Add the final set point to ensure the trajectory ends correctly
    trajectory.append(set_points[-1])

    return trajectory


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    camera: Camera = scene["camera"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller_left = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )
    diff_ik_controller_right = DifferentialIKController(
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
    robot_entity_cfg_right.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        left_ee_jacobi_idx = robot_entity_cfg_left.body_ids[0] - 1
        right_ee_jacobi_idx = robot_entity_cfg_right.body_ids[0] - 1
    else:
        left_ee_jacobi_idx = robot_entity_cfg_left.body_ids[0]
        right_ee_jacobi_idx = robot_entity_cfg_right.body_ids[0]

    teleop_interface = DualArmSe3Keyboard(
        pos_sensitivity=0.05 * args_cli.sensitivity,
        rot_sensitivity=0.05 * args_cli.sensitivity,
    )
    # add teleoperation key for env reset
    teleop_interface.add_callback("B", sim.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    sim.reset()
    teleop_interface.reset()

    # Create replicator writer
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output", "data"
    )
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor(
        [[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device
    )
    camera_targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    camera.set_world_poses_from_view(camera_positions, camera_targets)

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id
    npoints = 1024
    saved_sampled_points = []

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui():
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():

        delta_pose, gripper_command = teleop_interface.advance()
        delta_pose = delta_pose.astype("float32")
        # # convert to torch
        # delta_pose = torch.tensor(delta_pose, device=sim.device)

        # 左臂控制
        left_joint_pos_des = control_single_arm(
            robot,
            robot_entity_cfg_left,
            diff_ik_controller_left,
            left_ee_jacobi_idx,
            delta_pose[:6],  # 左臂的增量
        )

        # 右臂控制
        right_joint_pos_des = control_single_arm(
            robot,
            robot_entity_cfg_right,
            diff_ik_controller_right,
            right_ee_jacobi_idx,
            delta_pose[6:],  # 右臂的增量
        )

        # apply actions
        robot.set_joint_position_target(
            left_joint_pos_des, joint_ids=robot_entity_cfg_left.joint_ids
        )
        # print(left_joint_pos_des)
        robot.set_joint_position_target(
            right_joint_pos_des, joint_ids=robot_entity_cfg_right.joint_ids
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

        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        if False:
            print(camera)
            if "rgb" in camera.data.output.keys():
                print(
                    "Received shape of rgb image        : ",
                    camera.data.output["rgb"].shape,
                )
            if "distance_to_image_plane" in camera.data.output.keys():
                print(
                    "Received shape of depth image      : ",
                    camera.data.output["distance_to_image_plane"].shape,
                )
            if "normals" in camera.data.output.keys():
                print(
                    "Received shape of normals          : ",
                    camera.data.output["normals"].shape,
                )
            if "semantic_segmentation" in camera.data.output.keys():
                print(
                    "Received shape of semantic segm.   : ",
                    camera.data.output["semantic_segmentation"].shape,
                )
            if "instance_segmentation_fast" in camera.data.output.keys():
                print(
                    "Received shape of instance segm.   : ",
                    camera.data.output["instance_segmentation_fast"].shape,
                )
            if "instance_id_segmentation_fast" in camera.data.output.keys():
                print(
                    "Received shape of instance id segm.: ",
                    camera.data.output["instance_id_segmentation_fast"].shape,
                )
            print("-------------------------------")

        # Derive pointcloud from camera at camera_index
        pointcloud = create_pointcloud_from_depth(
            intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
            depth=camera.data.output["distance_to_image_plane"][camera_index],
            position=camera.data.pos_w[camera_index],
            orientation=camera.data.quat_w_ros[camera_index],
            device=sim.device,
        )

        # In the first few steps, things are still being instanced and Camera.data
        # can be empty. If we attempt to visualize an empty pointcloud it will crash
        # the sim, so we check that the pointcloud is not empty.
        right_ee_pose = robot.data.body_state_w[
            :, robot_entity_cfg_right.body_ids[0], 0:7
        ]
        left_ee_pose = robot.data.body_state_w[
            :, robot_entity_cfg_left.body_ids[0], 0:7
        ]

        if pointcloud.size()[0] > 0:
            pointcloud = pointcloud[pointcloud[:, 2] >= 0.01]
            sampled_points = farthest_point_sample(pointcloud, npoints)
            print("Sampled point cloud shape:", sampled_points.shape)

            # 获取关节数据
            left_joint_pos = (
                robot.data.joint_pos[:, robot_entity_cfg_left.joint_ids].cpu().numpy()
            )
            left_joint_vel = (
                robot.data.joint_vel[:, robot_entity_cfg_left.joint_ids].cpu().numpy()
            )
            right_joint_pos = (
                robot.data.joint_pos[:, robot_entity_cfg_right.joint_ids].cpu().numpy()
            )
            right_joint_vel = (
                robot.data.joint_vel[:, robot_entity_cfg_right.joint_ids].cpu().numpy()
            )

            # 获取末端执行器的姿态（转为 NumPy 数组）
            right_ee_pose = right_ee_pose.cpu().numpy()
            left_ee_pose = left_ee_pose.cpu().numpy()

            # 确保数据是 1D
            left_joint_pos = (
                left_joint_pos[0] if left_joint_pos.shape[0] == 1 else left_joint_pos
            )
            left_joint_vel = (
                left_joint_vel[0] if left_joint_vel.shape[0] == 1 else left_joint_vel
            )
            right_joint_pos = (
                right_joint_pos[0] if right_joint_pos.shape[0] == 1 else right_joint_pos
            )
            right_joint_vel = (
                right_joint_vel[0] if right_joint_vel.shape[0] == 1 else right_joint_vel
            )
            right_ee_pose = (
                right_ee_pose[0] if right_ee_pose.shape[0] == 1 else right_ee_pose
            )
            left_ee_pose = (
                left_ee_pose[0] if left_ee_pose.shape[0] == 1 else left_ee_pose
            )

            # 保存到 CSV 文件（追加模式）
            save_csv_path = os.path.join(output_dir, "data1.csv")
            file_exists = os.path.isfile(save_csv_path)
            with open(save_csv_path, mode="a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)

                # 如果文件不存在，写入表头
                if not file_exists:
                    header = [
                        "Sampled_Point",
                        "Left_Joint_Pos",
                        "Left_Joint_Vel",
                        "Right_Joint_Pos",
                        "Right_Joint_Vel",
                        "Left_EE_Pose",
                        "Right_EE_Pose",
                    ]
                    csv_writer.writerow(header)
                # 写入新的一行
                csv_writer.writerow(
                    [
                        sampled_points,
                        left_joint_pos.tolist(),
                        left_joint_vel.tolist(),
                        right_joint_pos.tolist(),
                        right_joint_vel.tolist(),
                        left_ee_pose.tolist(),
                        right_ee_pose.tolist(),
                    ]
                )

            print(f"Appended data to CSV: {save_csv_path}")


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
