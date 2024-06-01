import argparse
import os
from pathlib import Path

import init_path
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import h5py
import liblzfse
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from orion.utils.misc_utils import *

class R3DDatasetConverter:
    def __init__(
        self,
        r3d_path,
        camera_name="front_camera",
        depth=True,
        video=True,
        quiet=False,
        rotate=False,
        front_camera=False,
        cut_head=112,
        cut_tail=78,
    ):
        self.camera_name = "front_camera"
        self.r3d_path = r3d_path
        self.depth = depth
        self.video = video
        self.quiet = quiet
        self.rotate = rotate
        self.front_camera = front_camera
        self.cut_head = cut_head
        self.cut_tail = cut_tail

    def load_depth(self, filepath):
        with open(filepath, "rb") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

        # if not self.front_camera:
        #     depth_img = depth_img.reshape((256, 192))
        # else:
        #     depth_img = depth_img.reshape((640, 480))
        depth_img = depth_img.reshape((640, 480))
        depth_img = depth_img * 1000
        # print("shape", depth_img.shape)
        depth_img = depth_img.astype(dtype=np.uint16)
        return depth_img

    # def get_video(self, folder, data_type):
    #     if (data_type == 'depth'):
    #         typename = 'png'
    #         in_folder = os.path.join(folder, 'depth')
    #         out = os.path.join(folder, 'depth.mp4')
    #     else:
    #         typename = 'jpg'
    #         in_folder = os.path.join(folder, 'color')
    #         out = os.path.join(folder, 'color.mp4')
    #     os.system('ffmpeg -loglevel quiet -threads 2 -y -r 30 -i %s/%%d.%s %s' % (in_folder, typename, out))

    def run(self, folder=None):
        if not os.path.exists(self.r3d_path):
            print("ERROR: Wrong path!")
            return
        if folder is None:
            folder = self.r3d_path[:-4]

        os.system("unzip -q -n " + self.r3d_path + " -d " + folder)
        os.system("rm " + self.r3d_path[:-4] + "/sound.aac")
        os.system("rm " + self.r3d_path[:-4] + "/icon")

        with open(os.path.join(folder, "metadata")) as f:
            metadata = json.load(f)
        K = np.array(metadata["K"]).reshape(3, 3).T

        filename = folder.split("/")[-1]
        # target_size = (480, 640) if self.front_camera else (192, 256)
        # scale_w, scale_h = (
        #     (1.0, 1.0) if self.front_camera else (target_size[1] / 1440, target_size[0] / 1920)
        # )

        parent_folder = os.path.dirname(folder)
        with h5py.File(os.path.join(parent_folder, f"{filename}_demo.hdf5"), "w") as f:
            # Process and save data here as in the original script
            # The detailed implementation for data processing and saving has been omitted for brevity
            grp = f.create_group("data")

            ep_grp = grp.create_group("human_demo")
            if os.path.isdir(folder):
                rgbd_folder = os.path.join(folder, "rgbd")
                if self.depth:
                    for j in os.listdir(rgbd_folder):
                        if j[-6:] == ".depth":
                            depth_path = os.path.join(rgbd_folder, j)
                            # os.system('lzfse -decode -i '+depth_path+' -o '+depth_path+'_new')
                            if self.depth:
                                depth_img = self.load_depth(depth_path)
                                # cv2.imwrite(depth_path[:-6]+'.png', depth_img)
                                # o3d.io.write_image(depth_path[:-6]+'.png', depth_img)
                                depth_img = o3d.geometry.Image(depth_img)
                                o3d.io.write_image(depth_path[:-6] + ".png", depth_img)
                    os.system(f"mkdir {folder}/depth; mv {folder}/rgbd/*.png {folder}/depth")
                    os.system(f"mkdir {folder}/color; mv {folder}/rgbd/*.jpg {folder}/color")
                    if not self.quiet:
                        print("Depth data saved at %s/depth" % (folder))

                folder_path = Path(os.path.join(folder, "depth"))
                files = folder_path.glob("*.png")
                # Convert to list of strings
                depth_file_list = [str(file) for file in files]
                # Sort the list by converting the filenames to integers
                sorted_depth_files = sorted(depth_file_list, key=lambda x: int(Path(x).stem))

                folder_path = Path(os.path.join(folder, "color"))
                files = folder_path.glob("*.jpg")
                # Convert to list of strings
                color_file_list = [str(file) for file in files]
                # Sort the list by converting the filenames to integers
                sorted_color_files = sorted(color_file_list, key=lambda x: int(Path(x).stem))

                assert len(sorted_depth_files) == len(sorted_color_files)

                color_seq = []
                depth_seq = []

                # if self.skip_frames > 1:
                #     sorted_depth_files = sorted_depth_files[::self.skip_frames]
                #     sorted_color_files = sorted_color_files[::self.skip_frames]
                for depth_file, color_file in zip(sorted_depth_files, sorted_color_files):
                    depth_img = np.ascontiguousarray(cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH))
                    color_img = np.ascontiguousarray(cv2.imread(color_file)[..., ::-1])

                    color_seq.append(color_img)
                    depth_seq.append(depth_img)

                color_seq = np.stack(color_seq)
                depth_seq = np.stack(depth_seq)
                if self.rotate:
                    if not self.front_camera:
                        color_seq = np.rot90(color_seq, k=1, axes=(1, 2))
                        depth_seq = np.rot90(depth_seq, k=1, axes=(1, 2))
                    else:
                        color_seq = np.rot90(color_seq, k=3, axes=(1, 2))
                        depth_seq = np.rot90(depth_seq, k=3, axes=(1, 2))
                obs_grp = ep_grp.create_group("obs")

                color_seq = color_seq[self.cut_head : -self.cut_tail]
                depth_seq = depth_seq[self.cut_head : -self.cut_tail]

                obs_grp.create_dataset("agentview_rgb", data=color_seq)
                obs_grp.create_dataset("agentview_depth", data=depth_seq)
                print(f"Color shape: {color_seq.shape}")

                if self.video:
                    with VideoWriter(
                        parent_folder, save_video=True, fps=30, video_name="color.mp4"
                    ) as video_writer:
                        for i in range(len(color_seq)):
                            video_writer.append_image(color_seq[i])
                    video_writer.save(f"{filename}_color.mp4", bgr=False)

            data_config = {}
            intrinsics = {
                "fx": K[0, 0],
                "fy": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
            }
            data_config["intrinsics"] = {'front_camera': intrinsics}
            data_config["extrinsics"] = {
                self.camera_name: {"translation": [0, 0, 0], "rotation": np.eye(3).tolist()}
            }
            grp.attrs["data_config"] = json.dumps(data_config)
            print("Dataset saved in ", os.path.join(parent_folder, f"{filename}_demo.hdf5"))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--r3d_folder", type=str)
    parser.add_argument("--target_folder", type=str)
    args = parser.parse_args()

    dataset_converter = R3DDatasetConverter(
        r3d_path="",
        depth=True,
        video=True,
        quiet=False,
        rotate=True,
        front_camera=True,
    )
    r3d_folder = args.r3d_folder
    count = 0
    for r3d_file in Path(r3d_folder).rglob("*.r3d"):
        dataset_converter.r3d_path = str(r3d_file)

        target_folder = os.path.join(args.target_folder, f"{count}")
        os.makedirs(target_folder, exist_ok=True)
        dataset_converter.run(folder=target_folder)
        count += 1


if __name__ == "__main__":
    main()