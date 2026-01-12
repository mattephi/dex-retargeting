"""
Real-time retargeting from MetaQuest hand tracking data.

This script receives hand tracking data from a MetaQuest device over the network
and retargets it to a robot hand in real-time using SAPIEN visualization.

Requires the meta-teleop-client package and a running MetaQuest app streaming
hand tracking data.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

# Add meta-teleop-client to path
meta_teleop_path = Path(__file__).absolute().parent.parent.parent.parent / "meta-teleop-client" / "src"
sys.path.insert(0, str(meta_teleop_path))

from meta_teleop import MetaTeleopClient, CoordinateSystem

# Mapping from VR Quest 3 joint indices to MediaPipe 21-joint format
# VR has 26 joints (OpenXR), MediaPipe has 21
VR_TO_MEDIAPIPE_INDICES = np.array([
    1,   # WRIST <- Wrist
    2,   # THUMB_CMC <- ThumbMetacarpal
    3,   # THUMB_MCP <- ThumbProximal
    4,   # THUMB_IP <- ThumbDistal
    5,   # THUMB_TIP <- ThumbTip
    7,   # INDEX_MCP <- IndexProximal
    8,   # INDEX_PIP <- IndexIntermediate
    9,   # INDEX_DIP <- IndexDistal
    10,  # INDEX_TIP <- IndexTip
    12,  # MIDDLE_MCP <- MiddleProximal
    13,  # MIDDLE_PIP <- MiddleIntermediate
    14,  # MIDDLE_DIP <- MiddleDistal
    15,  # MIDDLE_TIP <- MiddleTip
    17,  # RING_MCP <- RingProximal
    18,  # RING_PIP <- RingIntermediate
    19,  # RING_DIP <- RingDistal
    20,  # RING_TIP <- RingTip
    22,  # PINKY_MCP <- LittleProximal
    23,  # PINKY_PIP <- LittleIntermediate
    24,  # PINKY_DIP <- LittleDistal
    25,  # PINKY_TIP <- LittleTip
])

# Coordinate transform from FLU (Forward-Left-Up) to MANO convention
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3D key points.
    Same as SingleHandDetector.estimate_frame_from_hand_points.
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index mcp, middle mcp

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram-Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


def extract_joint_positions(data, hand_type: str) -> Optional[np.ndarray]:
    """
    Extract and transform joint positions from MetaQuest finger joints data.

    Args:
        data: FingerJointsData from MetaQuest
        hand_type: "Left" or "Right"

    Returns:
        Joint positions in MANO format (21, 3) or None if hand not tracked
    """
    hand_joints = data.left if hand_type == "Left" else data.right

    if not hand_joints.is_tracked:
        return None

    # Get positions in FLU coordinate system (standard robotics convention)
    positions = hand_joints.positions(CoordinateSystem.FLU)  # (26, 3)

    # Map VR 26 joints to MediaPipe 21 joints
    joint_pos = positions[VR_TO_MEDIAPIPE_INDICES]  # (21, 3)

    # Center on wrist
    joint_pos = joint_pos - joint_pos[0:1, :]

    # Estimate frame and transform coordinates
    mediapipe_wrist_rot = estimate_frame_from_hand_points(joint_pos)
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
    joint_pos = joint_pos @ mediapipe_wrist_rot @ operator2mano

    return joint_pos


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    connection_timeout: float = 10.0,
):
    """
    Receives real-time hand tracking data from MetaQuest and retargets to a robot hand.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        connection_timeout: Timeout in seconds for connecting to MetaQuest.
    """
    # Setup retargeting
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type_str = "Right" if hand_type == HandType.right else "Left"

    # Setup SAPIEN scene
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name_str = filepath.stem
    loader.load_multiple_collisions_from_file = True

    if "ability" in robot_name_str:
        loader.scale = 1.5
    elif "dclaw" in robot_name_str:
        loader.scale = 1.25
    elif "allegro" in robot_name_str:
        loader.scale = 1.4
    elif "shadow" in robot_name_str:
        loader.scale = 0.9
    elif "bhand" in robot_name_str:
        loader.scale = 1.5
    elif "leap" in robot_name_str:
        loader.scale = 1.4
    elif "svh" in robot_name_str:
        loader.scale = 1.5

    if "glb" not in robot_name_str:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name_str:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # Connect to MetaQuest
    logger.info("Connecting to MetaQuest...")
    with MetaTeleopClient() as client:
        logger.info("Waiting for MetaQuest channels...")
        client.wait_for_channels(timeout=connection_timeout)
        logger.info(f"Connected to {client.device_name} ({client.device_model})")

        channel = client['finger_joints']
        logger.info(f"Receiving finger joints at ~{channel.stats.target_hz:.0f} Hz")
        logger.info("Starting real-time retargeting. Press 'q' in viewer to quit.")

        while True:
            # Get latest finger joints data
            data = channel.last_packet

            if data is None:
                for _ in range(2):
                    viewer.render()
                continue

            joint_pos = extract_joint_positions(data, hand_type_str)

            if joint_pos is None:
                logger.warning(f"{hand_type_str} hand is not tracked.")
            else:
                retargeting_type_enum = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices

                if retargeting_type_enum == "POSITION":
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                qpos = retargeting.retarget(ref_value)
                robot.set_qpos(qpos[retargeting_to_sapien])

            for _ in range(2):
                viewer.render()

    logger.info("Disconnected from MetaQuest.")


if __name__ == "__main__":
    tyro.cli(main)
