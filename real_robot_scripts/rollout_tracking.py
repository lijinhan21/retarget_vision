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
import cv2
import numpy as np
from easydict import EasyDict
from PIL import Image
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
import init_path
from orion.utils.misc_utils import VideoWriter, get_first_frame_annotation
from orion.utils.ik_utils import IKWrapper
from orion.utils.real_robot_utils import RealRobotObsProcessor
from orion.utils.misc_utils import read_from_runtime_file, finish_experiments, add_palette_on_mask, overlay_xmem_mask_on_image, get_palette
from orion.algos.xmem_tracker import XMemTracker

from tqdm import tqdm

def main():
    runtime_folder, rollout_folder, human_video_annotation_path = read_from_runtime_file()

    tmp_annotation_path = os.path.join(rollout_folder, "tmp_annootation.png")
    if not os.path.exists(tmp_annotation_path):
        first_frame, tmp_annotation = get_first_frame_annotation(human_video_annotation_path)        
    rollout_video_name = os.path.join(runtime_folder, "rollout.mp4")
    assert(os.path.exists(rollout_video_name)), "seems like you haven't done rollout yet. error when loading" + rollout_video_name
    videocap = cv2.VideoCapture(rollout_video_name)

    success, image = videocap.read()
    record_images = []

    while success: 
        record_images.append(image)
        success, image = videocap.read()

    assert(len(record_images) > 0)
    record_images = record_images[::10]
    tmp_annotation_path = os.path.join(runtime_folder, "tmp_annotation.png")
    if not os.path.exists(tmp_annotation_path):
        _, tmp_annotation = get_first_frame_annotation(rollout_folder)
    else:
        tmp_annotation = np.array(Image.open(tmp_annotation_path))

    device = "cuda:0"
    xmem_tracker = XMemTracker(xmem_checkpoint=f'third_party/xmem_checkpoints/XMem.pth', device=device)
    xmem_tracker.clear_memory()        
    masks = xmem_tracker.track_video(record_images, tmp_annotation)

    last_img = record_images[-1]
    last_annotation = masks[-1]
    
    cv2.imwrite(os.path.join(runtime_folder, "tmp.jpg"), last_img)
    new_mask = Image.fromarray(last_annotation)
    new_mask.putpalette(get_palette())
    new_mask.save(tmp_annotation_path)
    
    with VideoWriter(runtime_folder, "annotation_video.mp4", save_video=True) as video_writer:
        for rgb_img, mask in tqdm(zip(record_images, masks), total=len(record_images)):
            overlay_img = overlay_xmem_mask_on_image(rgb_img, mask, rgb_alpha=0.4)
            video_writer.append_image(overlay_img)

    count = 0
    while os.path.exists(os.path.join(runtime_folder, f"rollout_{count}.mp4")):
        count += 1
    os.rename(rollout_video_name, os.path.join(runtime_folder, f"rollout_{count}.mp4"))
    # cv2.imwrite(os.path.join(rollout_folder, "frame.jpg"), record_images[-1])
    

if __name__ == "__main__":
    main()