"""
A simple user interface for XMem
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import sys
import json

import numpy as np
import argparse
import shutil
import torch

import matplotlib.pyplot as plt
from easydict import EasyDict
import init_path
from orion.utils.misc_utils import (
    load_first_frame_from_hdf5_dataset, 
    export_video_from_hdf5_dataset,
    load_first_frame_from_human_hdf5_dataset, 
    export_video_from_human_hdf5_dataset,
    overlay_xmem_mask_on_image
    )
from orion.algos.grounded_sam_wrapper import GroundedSamWrapper
import argparse


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="A simple user interface for XMem")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--demo", help="Path to a demo file")
    parser.add_argument("--rollout", help="Path to a rollout file")
    parser.add_argument("--human_demo", default="datasets/iphones/long_horizon_boat/iphone_long_horizon_boat_18_demo.hdf5", help="Path to a human demo file")
    parser.add_argument("--text", nargs="+", help="List of text")
    return parser.parse_args()

def main():
    args = parse_args()

    wrapper = GroundedSamWrapper()
    args.human_demo = os.path.join("datasets/iphones", args.human_demo)

    mode = ""
    if args.image is not None:
        mode = "image"
    elif args.video is not None:
        mode = "video"
    elif args.demo is not None:
        mode = "demo"
    elif args.human_demo is not None:
        mode = "human_demo"
    elif args.rollout is not None:
        mode = "rollout"

    annotation_folder = f"annotations/{mode}"
    tmp_folder = "tmp_images"

    if mode == "image":
        annotation_path = os.path.join(annotation_folder, args.image.split("/")[-1].split(".")[0])
    elif mode == "video":
        annotation_path = os.path.join(annotation_folder, args.video.split("/")[-1].split(".")[0])
    elif mode == "demo":
        annotation_path = os.path.join(annotation_folder, args.demo.split("/")[-1].split(".")[0])
    elif mode == "human_demo":
        annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])
    elif mode == "rollout":
        annotation_path = os.path.join(annotation_folder, args.rollout.split("/")[-1].split(".")[0])

    tmp_path = tmp_folder


    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    if mode == "image":
        first_frame = cv2.imread(args.image)
    elif mode == "rollout":
        first_frame = cv2.imread(args.rollout)
    elif mode == "video":
        _, first_frame = cv2.VideoCapture(args.video).read()
    elif mode == "demo":
        first_frame = load_first_frame_from_hdf5_dataset(args.demo, demo_idx=args.demo_idx, bgr=True)
    elif mode == "human_demo":
        first_frame = load_first_frame_from_human_hdf5_dataset(args.human_demo, bgr=True)

    cv2.imwrite(os.path.join(os.path.join(tmp_path, "images", "frame.jpg")), first_frame)

    args.images = tmp_path
    args.workspace = tmp_path

    print(args.images, tmp_folder)

    # launch_gui(args)
    print("Annotating the image with text input: ", args.text)

    final_mask_image = wrapper.segment(first_frame, args.text)
    os.makedirs(os.path.join(tmp_path, "masks"), exist_ok=True)
    final_mask_image.save(os.path.join(tmp_path, "masks", "frame.png"))
    overlay_image = overlay_xmem_mask_on_image(first_frame, np.array(final_mask_image), use_white_bg=True, rgb_alpha=0.3)

    try:
        plt.imshow(overlay_image)
        plt.show()
    except:
        pass

    # copy a image from a folder to another
    shutil.copyfile(os.path.join(tmp_path, "images", "frame.jpg"), os.path.join(annotation_path, "frame.jpg"))
    shutil.copyfile(os.path.join(tmp_path, "masks", "frame.png"), os.path.join(annotation_path, "frame_annotation.png"))
    print("Annotation saved to ", os.path.join(annotation_path, "frame_annotation.png"))
    with open(os.path.join(annotation_path, "config.json"), "w") as f:
        config_dict = {"mode": mode}
        if mode == "image":
            config_dict["original_file"] = args.image
        elif mode == "rollout":
            config_dict["original_file"] = args.rollout            
        elif mode == "video":
            config_dict["original_file"] = args.video
            config_dict["video_file"] = args.video
        elif mode == "demo":
            config_dict["original_file"] = args.demo
            video_path = export_video_from_hdf5_dataset(
                            dataset_name=args.demo, 
                            demo_idx=args.demo_idx,
                            video_path=annotation_path,
                            video_name=args.demo.split("/")[-1].split(".")[0],
                            bgr=True)
            config_dict["video_file"] = video_path
        elif mode == "human_demo":
            config_dict["original_file"] = args.human_demo
            video_path = export_video_from_human_hdf5_dataset(
                            dataset_name=args.human_demo, 
                            video_path=annotation_path,
                            video_name=args.human_demo.split("/")[-1].split(".")[0],
                            bgr=True)
            config_dict["video_file"] = video_path
        config_dict["text"] = args.text
        json.dump(config_dict, f)
    # remove the folder
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()

