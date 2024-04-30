import json
import os
from pathlib import Path
import argparse

import cv2
import h5py
import numpy as np
import shutil

def extract_frames(video_path, output_folder):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Frame counter
    frame_count = 0
    
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read

        # Save the frame as an image file
        frame_path = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

class R3DDatasetConverter:
    def __init__(
        self,
        filepath=None,
        camera_name="iphone",
        quiet=False,
        rotate=False,
        front_camera=False,
        cut_head=0,
        cut_tail=10,
    ):
        self.filepath = filepath
        self.camera_name = camera_name
        self.quiet = quiet
        self.rotate = rotate
        self.front_camera = front_camera
        self.cut_head = cut_head
        self.cut_tail = cut_tail

    def run(self, folder=None):
        if folder is None:
            folder = self.filepath[:-4]

        filename = folder.split("/")[-1]
        target_size = (640, 480) 
        scale_w, scale_h = (
            (1.0, 1.0)
        )

        parent_folder = os.path.dirname(folder)
        with h5py.File(os.path.join(parent_folder, f"{filename}_demo.hdf5"), "w") as f:
            # Process and save data here as in the original script
            # The detailed implementation for data processing and saving has been omitted for brevity
            grp = f.create_group("data")

            ep_grp = grp.create_group("human_demo")
            tmp_folder = "tmp_images"
            color_folder = os.path.join(tmp_folder, "color")
            os.makedirs(color_folder, exist_ok=True)
            
            # extract all the color frames from the video
            extract_frames(self.filepath, color_folder)

            # list all the color frames
            color_file_list = [str(file) for file in Path(color_folder).rglob("*.jpg")]
            # Sort the list by converting the filenames to integers
            sorted_color_files = sorted(color_file_list, key=lambda x: int(Path(x).stem))

            color_seq = []

            for color_file in sorted_color_files:
                color_img = np.ascontiguousarray(cv2.imread(color_file)[..., ::-1])

                if color_img.shape[0] > target_size[0]:
                    color_img = np.ascontiguousarray(
                        cv2.resize(color_img, target_size, interpolation=cv2.INTER_AREA)
                    )

                color_seq.append(color_img)
                cv2.imwrite(color_file, color_img)

            color_seq = np.stack(color_seq)
            if self.rotate:
                color_seq = np.rot90(color_seq, k=3, axes=(1, 2))
            obs_grp = ep_grp.create_group("obs")

            color_seq = color_seq[self.cut_head : -self.cut_tail]

            obs_grp.create_dataset("agentview_rgb", data=color_seq)
            print(f"Color shape: {color_seq.shape}")

            shutil.rmtree(tmp_folder)

            data_config = {}
            K = np.eye(3) # TODO: replace with actual number
            intrinsics = {
                "fx": K[0, 0] * scale_w,
                "fy": K[1, 1] * scale_h,
                "cx": K[0, 2] * scale_w,
                "cy": K[1, 2] * scale_h,
            }
            data_config["intrinsics"] = {self.camera_name: intrinsics}
            data_config["extrinsics"] = {
                self.camera_name: {"translation": [0, 0, 0], "rotation": np.eye(3).tolist()}
            }
            grp.attrs["data_config"] = json.dumps(data_config)
            print("Dataset saved in ", os.path.join(parent_folder, f"{filename}_demo.hdf5"))


# Example of how to use the class
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None)
    args = parser.parse_args()

    args.filepath = os.path.join("datasets/iphones", args.filepath)
    converter = R3DDatasetConverter(
        filepath=args.filepath,
        quiet=False,
        rotate=False,
        front_camera=True,
    )
    converter.run()