"""
Real-time hand retargeting from Quest VR headset using meta-teleop-client.

This example connects to a Quest headset via the meta-teleop-client library
and retargets the hand tracking data to a robot hand in real-time.

Usage:
    python quest_realtime_retargeting.py --robot-name rohand --hand-type left
"""

import time
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

# Import meta-teleop-client
from meta_teleop import MetaTeleopClient, CoordinateSystem


# Mapping from Quest OpenXR 26-joint format to MediaPipe 21-joint format
# Quest: 0=Palm, 1=Wrist, 2-5=Thumb, 6-10=Index, 11-15=Middle, 16-20=Ring, 21-25=Little
# MediaPipe: 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Little
QUEST_TO_MEDIAPIPE = np.array([
    1,   # 0: Wrist <- Quest Wrist
    2,   # 1: Thumb CMC <- Quest Thumb Metacarpal
    3,   # 2: Thumb MCP <- Quest Thumb Proximal
    4,   # 3: Thumb IP <- Quest Thumb Distal
    5,   # 4: Thumb Tip <- Quest Thumb Tip
    7,   # 5: Index MCP <- Quest Index Proximal
    8,   # 6: Index PIP <- Quest Index Intermediate
    9,   # 7: Index DIP <- Quest Index Distal
    10,  # 8: Index Tip <- Quest Index Tip
    12,  # 9: Middle MCP <- Quest Middle Proximal
    13,  # 10: Middle PIP <- Quest Middle Intermediate
    14,  # 11: Middle DIP <- Quest Middle Distal
    15,  # 12: Middle Tip <- Quest Middle Tip
    17,  # 13: Ring MCP <- Quest Ring Proximal
    18,  # 14: Ring PIP <- Quest Ring Intermediate
    19,  # 15: Ring DIP <- Quest Ring Distal
    20,  # 16: Ring Tip <- Quest Ring Tip
    22,  # 17: Little MCP <- Quest Little Proximal
    23,  # 18: Little PIP <- Quest Little Intermediate
    24,  # 19: Little DIP <- Quest Little Distal
    25,  # 20: Little Tip <- Quest Little Tip
])


def quest_to_mediapipe_joints(quest_positions: np.ndarray) -> np.ndarray:
    """Convert Quest 26-joint positions to MediaPipe 21-joint format."""
    return quest_positions[QUEST_TO_MEDIAPIPE]


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points.
    This normalizes hand orientation so retargeting works regardless of hand pose in world.

    Args:
        keypoint_3d_array: 21 keypoints in MediaPipe format, centered at wrist

    Returns:
        3x3 rotation matrix representing the wrist frame
    """
    assert keypoint_3d_array.shape == (21, 3)
    # Use wrist (0), index MCP (5), middle MCP (9) to define the hand plane
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


# Transformation matrices from operator frame to MANO convention
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


def main(
    robot_name: RobotName,
    hand_type: HandType,
    retargeting_type: RetargetingType = RetargetingType.vector,
    timeout: float = 10.0,
):
    """
    Real-time hand retargeting from Quest VR headset.

    Connects to a Quest headset via meta-teleop-client and retargets
    the hand tracking data to a robot hand visualization.

    Args:
        robot_name: The identifier for the robot (e.g., rohand, allegro, shadow).
        hand_type: Specifies which hand to track and retarget (left or right).
        retargeting_type: The type of retargeting algorithm to use.
        timeout: Timeout in seconds for waiting for Quest connection.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Loading retargeting config from {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    config = RetargetingConfig.load_from_file(config_path)

    # Setup SAPIEN scene
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

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

    # Scale adjustments for different robots
    scale_map = {
        "ability": 1.5, "dclaw": 1.25, "allegro": 1.4,
        "shadow": 0.9, "bhand": 1.5, "leap": 1.4, "svh": 1.5
    }
    for name, scale in scale_map.items():
        if name in robot_name_str:
            loader.scale = scale
            break

    # Use GLB variant for visualization
    if "glb" not in robot_name_str:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # Position adjustments for different robots
    pose_map = {
        "ability": [0, 0, -0.15], "shadow": [0, 0, -0.2], "dclaw": [0, 0, -0.15],
        "allegro": [0, 0, -0.05], "bhand": [0, 0, -0.2], "leap": [0, 0, -0.15],
        "svh": [0, 0, -0.13]
    }
    for name, pos in pose_map.items():
        if name in robot_name_str:
            robot.set_pose(sapien.Pose(pos))
            break

    # Joint name mapping
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # Select OPERATOR2MANO based on hand type
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == HandType.right else OPERATOR2MANO_LEFT

    # Connect to Quest
    logger.info("Connecting to Quest headset...")
    with MetaTeleopClient() as client:
        if not client.wait_for_channels(timeout=timeout):
            logger.error(f"No Quest device found within {timeout}s. Make sure the Quest app is running.")
            return

        logger.info(f"Connected to {client.device_name} ({client.device_model})")
        channel = client.get_channel('finger_joints')

        logger.info(f"Starting {hand_type.name} hand retargeting. Press Ctrl+C to quit.")

        try:
            with channel.fast_receiver() as receiver:
                for packet, header in receiver:
                    # Get hand data based on hand type
                    if hand_type == HandType.left:
                        hand_data = packet.left
                    else:
                        hand_data = packet.right

                    if not hand_data.is_tracked:
                        logger.debug(f"{hand_type.name.capitalize()} hand not tracked")
                        for _ in range(2):
                            viewer.render()
                        continue

                    # Get positions in FLU coordinate system (ROS convention)
                    quest_positions = hand_data.positions(CoordinateSystem.FLU)

                    # Convert to MediaPipe 21-joint format
                    joint_pos = quest_to_mediapipe_joints(quest_positions)

                    # Center at wrist (joint 0)
                    joint_pos = joint_pos - joint_pos[0:1, :]

                    # Estimate hand frame and transform to canonical orientation
                    wrist_frame = estimate_frame_from_hand_points(joint_pos)
                    joint_pos = joint_pos @ wrist_frame @ operator2mano

                    # Compute retargeting
                    retargeting_type_str = retargeting.optimizer.retargeting_type
                    indices = retargeting.optimizer.target_link_human_indices

                    if retargeting_type_str == "POSITION":
                        ref_value = joint_pos[indices, :]
                    else:
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

                    qpos = retargeting.retarget(ref_value)
                    robot.set_qpos(qpos[retargeting_to_sapien])

                    for _ in range(2):
                        viewer.render()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Retargeting stopped")


if __name__ == "__main__":
    tyro.cli(main)
