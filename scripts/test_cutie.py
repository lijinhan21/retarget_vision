import os
import argparse

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from tqdm import tqdm

import init_path
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from orion.utils.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter, overlay_xmem_mask_on_image, add_palette_on_mask, get_video_seq_from_annotation, get_palette


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default="aircraft_demo")
    return parser.parse_args()

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():

    args = parse_args()
    annotation_folder = args.annotation_folder
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    frames = get_video_seq_from_annotation(annotation_folder)
    first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)
    first_frame_annotation = Image.fromarray(first_frame_annotation)
    image_path = './examples/images/bike'
    # images = sorted(os.listdir(image_path))  # ordering is important
    # mask = Image.open('./examples/masks/bike/00000.png')

    mask = first_frame_annotation
    palette = get_palette() # mask.getpalette()
    objects = np.unique(np.array(mask))
    objects = objects[objects != 0].tolist()  # background "0" does not count as an object
    mask = torch.from_numpy(np.array(mask)).cuda()

    with VideoWriter(annotation_folder, "test_cutie_tracking_video.mp4", save_video=True) as video_writer:
        for ti, rgb_image in enumerate(tqdm(frames)):
            # image = Image.open(os.path.join(image_path, image_name))
            image = Image.fromarray(rgb_image[..., ::-1])
            image = to_tensor(image).cuda().float()

            if ti == 0:
                output_prob = processor.step(image, mask, objects=objects)
            else:
                output_prob = processor.step(image)

            # convert output probabilities to an object mask
            mask = processor.output_prob_to_mask(output_prob)

            # visualize prediction
            mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
            mask.putpalette(palette)
            # mask.show()  # or use mask.save(...) to save it somewhere
            overlay_img = overlay_xmem_mask_on_image(rgb_image, np.array(mask), rgb_alpha=0.4)
            video_writer.append_image(overlay_img)

main()