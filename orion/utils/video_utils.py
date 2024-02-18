

import sys
import cv2
import argparse 
import os 
import torchvision

import numpy as np
try:
    import pyzed.sl as sl
except:
    print("ZED SDK not found. Please install it from https://www.stereolabs.com/developers/")

from copy import deepcopy


def read_from_svo(filepath):
    # Create a Camera object
    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)  #Set init parameter to run from 

    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE 

    cam = sl.Camera()
    status = cam.open(init)

    resolution = cam.get_camera_information().camera_configuration.resolution
    low_resolution = sl.Resolution(min(720,resolution.width) * 2, min(404,resolution.height))
    svo_image = sl.Mat(min(720,resolution.width) * 2,min(404,resolution.height), sl.MAT_TYPE.U8_C4, sl.MEM.CPU)


    runtime = sl.RuntimeParameters()

    mat = sl.Mat()

    fps = cam.get_camera_information().camera_configuration.fps
    nb_frames = cam.get_svo_number_of_frames()

    stereo_images = []
    for i in range(nb_frames):
        err = cam.grab(runtime)

        cam.retrieve_image(svo_image,sl.VIEW.SIDE_BY_SIDE,sl.MEM.CPU,low_resolution) #retrieve image left and right
        svo_position = cam.get_svo_position()
        frame = deepcopy(svo_image.get_data())[..., :3]
        stereo_images.append(frame)
    cam.close()
    return {"stereo_images":stereo_images, 
            "fps":fps, 
            "nb_frames":nb_frames, 
            "resolution":resolution, 
            "low_resolution":low_resolution}


# read image sequences from a mp4 video
def read_from_mp4(video_file):
    frames, _, meta_data = torchvision.io.read_video(video_file)
    frames = frames.detach().cpu().numpy().astype(np.uint8)
    return {"images": frames}
