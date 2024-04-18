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
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_demo", default="iphone_front_boat/iphone_front_boat_demo.hdf5", help="Path to a human demo file")
    return parser.parse_args()

def main():
    args = parse_args()

    wrapper = GroundedSamWrapper()
    args.human_demo = os.path.join("datasets/iphones", args.human_demo)

    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    tmp_folder = "tmp_images"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    tmp_path = tmp_folder

    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    first_frame = load_first_frame_from_human_hdf5_dataset(args.human_demo, bgr=True)

    cv2.imwrite(os.path.join(os.path.join(tmp_path, "images", "frame.jpg")), first_frame)

    with open(os.path.join(annotation_path, "text_description.json"), "r") as f:
        text_description = json.load(f)["objects"]
    print("Annotating the image with text input: ", text_description)

    final_mask_image = wrapper.segment(first_frame, text_description)
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
        config_dict["original_file"] = args.human_demo
        video_path = export_video_from_human_hdf5_dataset(
                        dataset_name=args.human_demo, 
                        video_path=annotation_path,
                        video_name=args.human_demo.split("/")[-1].split(".")[0],
                        bgr=True)
        config_dict["video_file"] = video_path
        config_dict["text"] = text_description
        json.dump(config_dict, f)
    # remove the folder
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()

