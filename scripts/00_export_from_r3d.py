import numpy as np
import matplotlib.pyplot as plt
import cv2
import struct
import os
import sys
import json
import argparse
import liblzfse
import open3d as o3d

import h5py

from pathlib import Path

def load_depth(filepath, front_camera=False):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    if not front_camera:
        depth_img = depth_img.reshape((256, 192))
    else:
        depth_img = depth_img.reshape((640, 480))
    depth_img = depth_img * 1000
    depth_img = depth_img.astype(dtype= np.uint16)
    return depth_img

def depth2image(filename, nx, ny):
    f = open(filename, "rb")
    img = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            data = f.read(4)
            elem = struct.unpack("f", data)[0]
            img[i][j] = elem
    f.close()
    return img

def get_video(folder, typename):
    if (typename == 'depth'):
        typename = 'png'
        in_folder = os.path.join(folder, 'depth')
        out = os.path.join(folder, 'depth.mp4')
    else:
        in_folder = os.path.join(folder, 'color')
        out = os.path.join(folder, 'color.mp4')
    os.system('ffmpeg -loglevel quiet -threads 2 -y -r 30 -i %s/%%d.%s %s'%(in_folder, typename, out))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r3d-path', required=True)
    parser.add_argument('--depth', action='store_const', const='depth')
    parser.add_argument('--video', action='store_const', const='video')
    parser.add_argument('--quiet', action='store_const', const='quiet')
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--front-camera', action='store_true')
    parser.add_argument('--cut-head', type=int, default=10)
    parser.add_argument('--cut-tail', type=int, default=40)
    # parser.add_argument('--skip-frames', type=int, default=1)
    args = parser.parse_args()

    # read intrinsics as follows
    # with open("/home/yifengz/workspace/in_context_open_world_manipulation/RGBD-Video-iPhone12Pro/r3d_files/test_home/metadata", "r") as f:
    #     data = json.load(f)
    # np.array(data["K"]).reshape(3, 3).T
    if(not os.path.exists(args.r3d_path)):
        print('ERROR: Wrong path!')
        return
    # if(os.listdir(args.r3d_path)==[]):
    #     print('ERROR: Empty folder!')
    #     return
    # unzip r3d
    # for r3d in os.listdir(args.r3d_path):
    # if(r3d[-4:] == '.r3d'):
    os.system('unzip -q -n '+ args.r3d_path +' -d '+args.r3d_path[:-4])
    os.system('rm '+args.r3d_path[:-4]+'/sound.aac')
    os.system('rm '+args.r3d_path[:-4]+'/icon')
    # decoding depth files
    # for i in os.listdir(args.r3d_path):
    folder = args.r3d_path[:-4]

    with open(os.path.join(folder, 'metadata'), "r") as f:
        metadata = json.load(f)
    K = np.array(metadata["K"]).reshape(3, 3).T

    filename = folder.split('/')[-1]

    target_size = (480, 640)
    # target_size = (192, 256)

    if not args.front_camera:
        scale_w = target_size[1] / 1440
        scale_h = target_size[0] / 1920
    else:
        scale_w = 1.
        scale_h = 1.
    parent_folder = os.path.dirname(folder)
    with h5py.File(os.path.join(parent_folder, f'{filename}_demo.hdf5'), 'w') as f:
        grp = f.create_group('data')

        ep_grp = grp.create_group("human_demo")
        if (os.path.isdir(folder)):
            rgbd_folder = os.path.join(folder, 'rgbd')
            if(args.depth):
                for j in os.listdir(rgbd_folder):
                    if(j[-6:] == '.depth'):
                        depth_path = os.path.join(rgbd_folder, j)
                        # os.system('lzfse -decode -i '+depth_path+' -o '+depth_path+'_new')
                        if(args.depth):
                            depth_img = load_depth(depth_path, front_camera=args.front_camera)
                            # cv2.imwrite(depth_path[:-6]+'.png', depth_img)
                            # o3d.io.write_image(depth_path[:-6]+'.png', depth_img)
                            depth_img = o3d.geometry.Image(depth_img)
                            o3d.io.write_image(depth_path[:-6]+'.png', depth_img)
                os.system('mkdir %s/depth; mv %s/rgbd/*.png %s/depth'%(folder, folder, folder))
                os.system('mkdir %s/color; mv %s/rgbd/*.jpg %s/color'%(folder, folder, folder))
                if (not args.quiet): print('Depth data saved at %s/depth'%(folder))
                    
            if(args.video):
                # rgb video
                get_video(folder, 'jpg')
                if (not args.quiet): print('RGB video saved at '+folder)
                if(args.depth):
                    # depth video
                    get_video(folder, 'depth')
                    if (not args.quiet): print('Depth video saved at '+folder)
        
            folder_path = Path(os.path.join(folder, 'depth'))
            files = folder_path.glob('*.png')
            # Convert to list of strings
            depth_file_list = [str(file) for file in files]
            # Sort the list by converting the filenames to integers
            sorted_depth_files = sorted(depth_file_list, key=lambda x: int(Path(x).stem))

            folder_path = Path(os.path.join(folder, 'color'))
            files = folder_path.glob('*.jpg')
            # Convert to list of strings
            color_file_list = [str(file) for file in files]
            # Sort the list by converting the filenames to integers
            sorted_color_files = sorted(color_file_list, key=lambda x: int(Path(x).stem))

            assert(len(sorted_depth_files) == len(sorted_color_files))
            
            color_seq = []
            depth_seq = []


            # if args.skip_frames > 1:
            #     sorted_depth_files = sorted_depth_files[::args.skip_frames]
            #     sorted_color_files = sorted_color_files[::args.skip_frames]
            for depth_file, color_file in zip(sorted_depth_files, sorted_color_files):
                depth_img = np.ascontiguousarray(cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH))
                color_img = np.ascontiguousarray(cv2.imread(color_file)[..., ::-1])
                # resized_color_img = np.ascontiguousarray(cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST))
                
                if color_img.shape[0] > target_size[0]:
                    color_img = np.ascontiguousarray(cv2.resize(color_img, target_size, interpolation=cv2.INTER_AREA))
                
                if depth_img.shape[0] > target_size[0]:
                    depth_img = np.ascontiguousarray(cv2.resize(depth_img, target_size, interpolation=cv2.INTER_NEAREST))

                color_seq.append(color_img)
                depth_seq.append(depth_img)

            color_seq = np.stack(color_seq)
            depth_seq = np.stack(depth_seq)
            if args.rotate:
                if not args.front_camera:
                    color_seq = np.rot90(color_seq, k=1, axes=(1,2))
                    depth_seq = np.rot90(depth_seq, k=1, axes=(1,2))
                else:
                    color_seq = np.rot90(color_seq, k=3, axes=(1,2))
                    depth_seq = np.rot90(depth_seq, k=3, axes=(1,2))
            obs_grp = ep_grp.create_group('obs')

            color_seq = color_seq[args.cut_head:-args.cut_tail]
            depth_seq = depth_seq[args.cut_head:-args.cut_tail]

            obs_grp.create_dataset('agentview_rgb', data=color_seq)
            obs_grp.create_dataset('agentview_depth', data=depth_seq)
            print(f"Color shape: {color_seq.shape}")

        data_config = {}

        intrinsics = {
            "fx": K[0, 0] * scale_w,
            "fy": K[1, 1] * scale_h,
            "cx": K[0, 2] * scale_w,
            "cy": K[1, 2] * scale_h,
        }
        data_config["intrinsics"] = {"camera_rs_0": intrinsics}
        data_config["extrinsics"] = {"camera_rs_0": {"translation": [0, 0, 0], "rotation": np.eye(3).tolist()}}
        grp.attrs["data_config"] = json.dumps(data_config)

    # visualize the image sequence
    for i in range(len(color_seq)):
        cv2.imshow('color', color_seq[i])
        cv2.waitKey(10)
    cv2.destroyAllWindows()


    print("Dataset saved in ", os.path.join(folder, f'{filename}_demo.hdf5'))

        
if __name__ == "__main__":
    main()