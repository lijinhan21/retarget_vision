import os
import argparse
import torchvision
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
import init_path
from orion.algos.xmem_tracker import XMemTracker
from orion.algos.cutie_tracker import CutieTracker
from orion.utils.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter, overlay_xmem_mask_on_image, add_palette_on_mask, get_video_seq_from_annotation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default="aircraft_demo")
    parser.add_argument("--tracker-type", type=str, default="cutie", choices=["xmem", "cutie"])    
    parser.add_argument("--multiple-annotations", action="store_true")
    return parser.parse_args()


def tracker_annotation(annotation_folder, tracker_type="xmem"):
    frames = get_video_seq_from_annotation(annotation_folder)

    first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)

    device = "cuda:0"
    if tracker_type == "xmem":
        tracker = XMemTracker(xmem_checkpoint=f'third_party/xmem_checkpoints/XMem.pth', device=device)
    elif tracker_type == "cutie":
        tracker = CutieTracker(device=device)

    tracker.clear_memory()

    masks = tracker.track_video(frames, first_frame_annotation)
    np.savez(os.path.join(annotation_folder, "masks.npz"), np.stack(masks, axis=0))

    with VideoWriter(annotation_folder, "xmem_annotation_video.mp4", save_video=True) as video_writer:
        for rgb_img, mask in tqdm(zip(frames, masks), total=len(frames)):
            overlay_img = overlay_xmem_mask_on_image(rgb_img, mask, rgb_alpha=0.4)
            video_writer.append_image(overlay_img)

def main():

    args = parse_args()

    if args.multiple_annotations:
        for annotation_folder in Path(args.annotation_folder).glob("*"):
            if not os.path.exists(annotation_folder / "masks.npz"):
                try:
                    tracker_annotation(annotation_folder, args.tracker_type)
                except:
                    print(f"Failed to process {annotation_folder}")
                    continue
            else:
                print(f"Skipping {annotation_folder} as masks.npz already exists")

    else:
        tracker_annotation(args.annotation_folder, args.tracker_type)

    

if __name__ == "__main__":
    main()