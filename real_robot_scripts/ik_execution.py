import argparse
import json
import os
import pickle
import torch
import threading
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict

from deoxys import config_root
from deoxys.experimental.motion_utils import follow_joint_traj, reset_joints_to, joint_interpolation_traj
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys_vision.utils.calibration_utils import load_default_extrinsics, load_default_intrinsics
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.utils.img_utils import save_depth_in_rgb

import time
from typing import Union

import numpy as np

from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)

import init_path

from orion.utils.misc_utils import VideoWriter, get_first_frame_annotation
from orion.utils.ik_utils import IKWrapper
from orion.utils.real_robot_utils import RealRobotObsProcessor, ImageCapturer
from orion.utils.misc_utils import read_from_runtime_file, finish_experiments
from orion.algos.xmem_tracker import XMemTracker

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", type=str)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    robot_config_parse_args(parser)
    return parser.parse_args()


def reset_joints_to_with_recording(
    robot_interface,
    start_joint_pos,
    controller_cfg: dict = None,
    timeout=7,
    gripper_open=False,
    image_capturer=None,
    no_record=False,
):
    assert type(start_joint_pos) is list or type(start_joint_pos) is np.ndarray
    if controller_cfg is None:
        controller_cfg = get_default_controller_config(controller_type="JOINT_POSITION")
    else:
        assert controller_cfg["controller_type"] == "JOINT_POSITION", (
            "This function is only for JOINT POSITION mode. You specified "
            + controller_cfg["controller_type"]
        )
        controller_cfg = verify_controller_config(controller_cfg)

    if gripper_open:
        gripper_action = -1
    else:
        gripper_action = 1
    if type(start_joint_pos) is list:
        action = start_joint_pos + [gripper_action]
    else:
        action = start_joint_pos.tolist() + [gripper_action]
    start_time = time.time()
    while True:
        if (
            robot_interface.received_states
            and robot_interface.check_nonzero_configuration()
        ):
            if (
                np.max(
                    np.abs(np.array(robot_interface.last_q) - np.array(start_joint_pos))
                )
                < 1e-3
            ):
                break
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )
        if image_capturer is not None and not no_record:
            image_capturer.record_obs()
        end_time = time.time()

        # Add timeout
        if end_time - start_time > timeout:
            break

def hand_move_to_position(robot_interface, device):
    controller_cfg = YamlConfig(os.path.join(config_root, "compliant-joint-impedance-controller.yml")).as_easydict()
    controller_type = "JOINT_IMPEDANCE"

    print("Please start moving ... ")

    while True:
        action, _ = input2action(
                        device=device,
                        controller_type="OSC_POSE",
                    )
        if action[-1] > 0:
            break
        action = list(robot_interface._state_buffer[-1].q) + [-1]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

    print("Moving ended")


def ik_execution(args, image_capturer: ImageCapturer):
    with open("experiments/runtime_T.json", "r") as f:
        args.motion_file = json.load(f)["file"]
    runtime_folder, rollout_folder, human_video_annotation_path = read_from_runtime_file()
    print(f"Run time folder: {runtime_folder}")
    print(f"Rollout folder: {rollout_folder}")
    print(f"Human video annotation path: {human_video_annotation_path}")

    # start camera for streaming images
    observation_cfg = YamlConfig("./configs/real_robot_observation_cfg.yml").as_easydict()
    observation_cfg.cameras = []
    for camera_ref in observation_cfg.camera_refs:
        assert_camera_ref_convention(camera_ref)
        camera_info = get_camera_info(camera_ref)

        observation_cfg.cameras.append(camera_info)

    image_capturer.get_last_obs()

    # Initialize Franka Interface
    robot_interface = FrankaInterface(os.path.join(config_root, args.interface_cfg))

    # Initialize spacemouse for terminating program in case of emergency
    device = SpaceMouse(vendor_id=9583, product_id=50734)
    device.start_control()

    # start reading robot states until they are received
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)

    # hand_move_to_position(robot_interface, device)
    controller_cfg = YamlConfig(os.path.join(config_root, "joint-impedance-controller.yml")).as_easydict()
    controller_type = "JOINT_IMPEDANCE"

    last_q = np.array(robot_interface.last_q)
    initial_q = last_q
    last_eef_mat, last_eef_pos = robot_interface.last_eef_rot_and_pos

    ik_wrapper = IKWrapper()
    data = torch.load(args.motion_file)
    R_seq = data["R_seq"]
    t_seq = data["t_seq"]
    target_interaction_centroid = torch.load(args.motion_file)["target_interaction_centroid"]
    target_interaction_points = torch.load(args.motion_file)["target_interaction_points"]

    logger.info("Starting IK ...")
    # joint_traj, new_T_seq, quat_seq  = ik_wrapper.ik_trajectory_from_T_seq(T_seq, last_q.tolist())

    z_rotation = data["z_rotation"]

    interaction_vector = target_interaction_points[1] - target_interaction_points[0]
    interaction_vector /= np.linalg.norm(interaction_vector)
    z_rotation_2 = np.arccos(
        np.dot(
            interaction_vector,
            np.array([0.0, 1.0, 0.0])
        )
    )
    z_rotation += z_rotation_2
    z_rotation = (np.pi - z_rotation) % np.pi
    if z_rotation > np.pi / 2:
        z_rotation = np.pi - z_rotation

    print("z rotation: ", z_rotation)

    z_rotation = 0
    pre_rotation_matrix = np.array([
        [np.cos(z_rotation), -np.sin(z_rotation), 0],
        [np.sin(z_rotation), np.cos(z_rotation), 0],
        [0, 0, 1]
    ])

    pre_pre_joint_seq, _, _ = ik_wrapper.ik_trajectory_from_R_t_seq(
        [pre_rotation_matrix],
        [np.zeros((1, 3))],
        last_q.tolist()
    )
    pre_pre_joint_seq = ik_wrapper.interpolate_dense_traj(pre_pre_joint_seq, minimal_displacement=0.03)
    start_q = pre_pre_joint_seq[-1].tolist()

    # start_q = last_q.tolist()
    num_points = 50
    pre_target_location = target_interaction_centroid + np.array([0, 0., 0.07])
    pre_joint_seq = ik_wrapper.ik_trajectory_to_target_position(
        np.array(pre_target_location),
        start_q,
        num_points=num_points)
    
    pre_joint_seq = np.concatenate([pre_pre_joint_seq, pre_joint_seq], axis=0)

    pre_joint_seq_2 = ik_wrapper.ik_trajectory_to_target_position(
        np.array(target_interaction_centroid),
        pre_joint_seq[-1].tolist(),
        num_points=num_points)

    reset_joint_positions = pre_joint_seq_2[-1].tolist()
    joint_seq, new_T_seq, quat_seq = ik_wrapper.ik_trajectory_from_R_t_seq(R_seq, t_seq, reset_joint_positions)
    # joint_seq = ik_wrapper.ik_trajectory_to_target_position(np.array([0.4, 0.2, 0.2]), reset_joint_positions)

    joint_seq = ik_wrapper.interpolate_dense_traj(joint_seq, minimal_displacement=0.01)
    joint_seq = np.array(joint_seq)

    # record_images = []
    # def record_image():
    #     color_imgs, depth_imgs = obs_processor.get_original_imgs()
    #     record_images.append(color_imgs[0])

    minimal_displacement = 0.05
    visualize_joints = []
    for joint_traj in [pre_joint_seq, pre_joint_seq_2, joint_seq]:
        joint_traj = ik_wrapper.interpolate_dense_traj(joint_traj, minimal_displacement=minimal_displacement)
        logger.info("Visualizing IK results ...")
        visualize_joints.append(joint_traj)
    visualize_joints = np.concatenate(visualize_joints, axis=0)
    ik_wrapper.visualize_joint_sequence(visualize_joints, fps=180)

    dirname = os.path.dirname(args.motion_file)
          
    valid_input = False
    while not valid_input:
        try:
            execute = input(f"Excute or not? (enter 0 - No or 1 - Yes)")
            execute = bool(int(execute))
            valid_input = True
        except ValueError:
            print("Please input 1 or 0!")
            continue

    if execute:
        for freespace_joint_seq in [pre_joint_seq, pre_joint_seq_2]:
            for joint in freespace_joint_seq:
                action = joint.tolist() + [-1.0]
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )

                image_capturer.record_obs()
            
        for _ in range(10):
            action = joint_seq[0].tolist() + [1.0]
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            image_capturer.record_obs()
        for joint in joint_seq:
            sp_action, grasp = input2action(
                        device=device,
                        controller_type="OSC_POSE",
                    )
            if sp_action is None:
                break
            
            action = joint.tolist() + [1.0]
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            image_capturer.record_obs()
        if sp_action is not None:
            for _ in range(10):
                action = joint_seq[-1].tolist() + [-1.0]
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                image_capturer.record_obs()

        last_eef_mat, last_eef_pos = robot_interface.last_eef_rot_and_pos
        post_position = last_eef_pos.reshape(1, 3) + np.array([0.0, 0.0, 0.09])
        last_q = np.array(robot_interface.last_q)

        post_joint_seq = ik_wrapper.ik_trajectory_to_target_position(
            np.array(post_position),
            last_q.tolist(),
            num_points=num_points)
        post_joint_seq = ik_wrapper.interpolate_dense_traj(post_joint_seq, minimal_displacement=minimal_displacement)            
        for post_joint in post_joint_seq:
            action = post_joint.tolist() + [-1.0]
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
            image_capturer.record_obs()

        if not args.no_reset:
            reset_joints_to_with_recording(robot_interface, initial_q, gripper_open=True, image_capturer=image_capturer, no_record=args.no_record)

    # record rollout video
    dirname = os.path.dirname(args.motion_file)
    with VideoWriter(runtime_folder, "rollout.mp4", save_video=not args.no_record) as video_writer:
        for rgb_img in image_capturer.record_images:
            video_writer.append_image(rgb_img)
    
    finish_experiments(runtime_folder, rollout_folder, human_video_annotation_path)  
    robot_interface.close()


def main():
    args = parse_args()
    image_capturer = ImageCapturer(record=not args.no_record)
    ik_execution(args, image_capturer)

if __name__ == "__main__":
    main()