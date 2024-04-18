import os
import shutil
import numpy as np

from PIL import Image
import cv2
from tqdm import tqdm

from orion.utils.misc_utils import VideoWriter, get_first_frame_annotation, read_from_runtime_file, finish_experiments, add_palette_on_mask, overlay_xmem_mask_on_image, get_palette
from orion.algos.xmem_tracker import XMemTracker

class RolloutTracker:
    def __init__(self):
        self.device = "cuda:0"
        self.xmem_tracker = XMemTracker(xmem_checkpoint='third_party/xmem_checkpoints/XMem.pth', device=self.device)

    def run(self):
        runtime_folder, rollout_folder, human_video_annotation_path = read_from_runtime_file()

        tmp_annotation_path = os.path.join(rollout_folder, "tmp_annotation.png")
        if not os.path.exists(tmp_annotation_path):
            first_frame, tmp_annotation = get_first_frame_annotation(human_video_annotation_path)
        rollout_video_name = os.path.join(runtime_folder, "rollout.mp4")
        assert os.path.exists(rollout_video_name), "seems like you haven't done rollout yet. error when loading " + rollout_video_name
        videocap = cv2.VideoCapture(rollout_video_name)

        success, image = videocap.read()
        record_images = []

        while success:
            record_images.append(image)
            success, image = videocap.read()

        assert len(record_images) > 0, "No images were recorded from the video."
        record_images = record_images[::10]
        tmp_annotation_path = os.path.join(runtime_folder, "tmp_annotation.png")
        if not os.path.exists(tmp_annotation_path):
            _, tmp_annotation = get_first_frame_annotation(rollout_folder)
        else:
            tmp_annotation = np.array(Image.open(tmp_annotation_path))

        self.xmem_tracker.clear_memory()
        masks = self.xmem_tracker.track_video(record_images, tmp_annotation)

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

        # copy this file to "annotation_{count}.mp4"
        shutil.copyfile(os.path.join(runtime_folder, "annotation_video.mp4"), os.path.join(runtime_folder, f"annotation_video_{count}.mp4"))
        count = 0
        while os.path.exists(os.path.join(runtime_folder, f"rollout_{count}.mp4")):
            count += 1
        os.rename(rollout_video_name, os.path.join(runtime_folder, f"rollout_{count}.mp4"))

# Usage example:
if __name__ == "__main__":
    tracker = RolloutTracker()
    tracker.run()