import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

import init_path
from orion.utils.misc_utils import *
from orion.algos.tap_segmentation import TAPSegmentation

from orion.utils.o3d_utils import (
    scene_pcd_fn, 
    O3DPointCloud, 
    estimate_rotation, 
    remove_outlier,
    load_reconstruction_info_from_human_demo, 
    project3Dto2D, 
    create_o3d_from_points_and_color,
    transform_point_clouds,
    filter_pcd)

def get_3d_points(input_image, depth, mask, filter=True, remove_outlier_kwargs={"nb_neighbors": 40, "std_ratio": 0.7}, downsample=True):
    # if object_id is None:
    #     masked_depth = self.input_depth * (self.input_annotation > 0).astype(np.float32)
    # else:
    #     masked_depth = self.input_depth * (self.input_annotation == object_id).astype(np.float32)
    masked_depth = depth * mask.astype(np.float32)
    camera_extrinsics = np.eye(4)
    camera_intrinsics = np.array([[909.83630371,   0.        , 651.97015381], [  0.        , 909.12280273, 376.37097168], [  0.        ,   0.        ,   1.        ]])
    pcd_points, pcd_colors = scene_pcd_fn(
        rgb_img_input=input_image,
        depth_img_input=masked_depth,
        extrinsic_matrix=camera_extrinsics,
        intrinsic_matrix=camera_intrinsics,
        downsample=downsample,
    )
    if filter:
        pcd_points, pcd_colors = remove_outlier(pcd_points, pcd_colors,
                                                **remove_outlier_kwargs)
    return pcd_points, pcd_colors

def determine_hand_object_contact_3d(img, depth, hand_masks, obj_masks, num_objs):
    """
    img: numpy array (H, W, 3), RGB image
    depth: numpy array (H, W), depth image
    hand_masks: list of numpy array (H, W), hand masks
    obj_masks: list of numpy array (H, W), object masks
    num_objs: int, number of objects
    """
    obj_pcds = []
    for i in range(num_objs):
        pcd_array, _ = get_3d_points(img, depth, obj_masks == i+1)
        pcd = create_o3d_from_points_and_color(pcd_array)
        obj_pcds.append(pcd)
    hand_pcds = []
    for i in range(2):
        pcd_array, _ = get_3d_points(img, depth, hand_masks[i, :, :, 0])
        pcd = create_o3d_from_points_and_color(pcd_array)
        hand_pcds.append(pcd)
    
    contacts = [-1, -1]
    for i in range(2):
        for j in range(num_objs):
            dists = hand_pcds[i].compute_point_cloud_distance(obj_pcds[j])
            intersections = [d for d in dists if d < 0.05]
            if len(intersections) > 600:
                contacts[i] = j
                break
    print("contacts =", contacts)
    return contacts

def determine_hand_object_contact_2d(img, hand_masks, obj_masks, num_objs):
    """
    img: numpy array (H, W, 3), RGB image
    hand_masks: list of numpy array (H, W), hand masks
    obj_masks: list of numpy array (H, W), object masks
    num_objs: int, number of objects
    """
    obj_mask_lst = []
    for i in range(num_objs):
        obj_mask_lst.append(obj_masks == i+1)
    hand_mask_lst = [hand_masks[0, :, :, 0], hand_masks[1, :, :, 0]]

    contacts = [-1, -1]
    for i in range(2):
        for j in range(num_objs):

            obj_pts = np.where(obj_mask_lst[j])
            hand_pts = np.where(hand_mask_lst[i])
            
            # sample no more than 10000 points
            obj_pts_idx = np.random.choice(len(obj_pts[0]), min(7000, len(obj_pts[0])), replace=False)
            hand_pts_idx = np.random.choice(len(hand_pts[0]), min(800, len(hand_pts[0])), replace=False)

            filtered_obj_pts = np.array([obj_pts[0][obj_pts_idx], obj_pts[1][obj_pts_idx]]).T
            filtered_hand_pts = np.array([hand_pts[0][hand_pts_idx], hand_pts[1][hand_pts_idx]]).T
            
            dists = np.concatenate([np.linalg.norm(filtered_obj_pts - hand_pt, axis=1) for hand_pt in filtered_hand_pts])
            intersections = np.sum(dists < 10)
            # print("number of intersections=", intersections)

            if intersections > 300:
                contacts[i] = j
                break
    # print("contacts =", contacts)
    return contacts

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument("--no-depth", action="store_true", default=False)
    args = parser.parse_args()

    tap_segmentation = TAPSegmentation()
    tap_segmentation.load(args.annotation_folder)

    # load video sequence
    video_seq = get_video_seq_from_annotation(args.annotation_folder)
    
    # load hand masks
    hand_mask_file = os.path.join(args.annotation_folder, 'hamer', "hand_masks.npz")
    hand_masks = np.load(hand_mask_file)['arr_0']
    print("shape of hand_masks =", hand_masks.shape)

    # load object masks
    mask_file = f"{args.annotation_folder}/masks.npz"
    if not os.path.exists(mask_file):
        raise ValueError(f"Mask file {mask_file} does not exist. You need to run XMem annotation first in order to proceed.")
    obj_masks = np.load(mask_file)['arr_0']

    num_objs = obj_masks[0].max()
    print("number of objects =", num_objs)
    
    if not args.no_depth:
        depth_seq = get_depth_seq_from_annotation(args.annotation_folder)

    all_segments_info = []
    for idx, seg in enumerate(tap_segmentation.temporal_segments.segments):
        
        all_res = []

        # sample several keyframes from each segment
        keyframe_indices = np.linspace(seg.start_idx, seg.end_idx, 15).astype(int)
        keyframe_indices = keyframe_indices[3:12]
        for i in range(len(keyframe_indices)):
            # check hand-object relationship in this frame
            img = video_seq[keyframe_indices[i]]
            hand_mask = hand_masks[keyframe_indices[i]]
            obj_mask = obj_masks[keyframe_indices[i]]
            if not args.no_depth:
                depth = depth_seq[keyframe_indices[i]]
                contacts = determine_hand_object_contact_3d(img, depth, hand_mask, obj_mask, num_objs)
            else:
                contacts = determine_hand_object_contact_2d(img, hand_mask, obj_mask, num_objs)
        
            all_res.append(contacts)

        num_left = {}
        num_right = {}
        for contact in all_res:
            if contact[0] not in num_left:
                num_left[contact[0]] = 0
            if contact[1] not in num_right:
                num_right[contact[1]] = 0
            num_left[contact[0]] += 1
            num_right[contact[1]] += 1

        # found bigest number of left and right hand
        max_left = 0
        max_right = 0
        segment_res = [-1, -1]
        for key, value in num_left.items():
            if value >= max_left:
                max_left = value
                segment_res[0] = key
        for key, value in num_right.items():
            if value >= max_right:
                max_right = value
                segment_res[1] = key

        print(f"Segment {idx} hand info: {segment_res}")

        all_segments_info.append(segment_res)
    
    with open(os.path.join(args.annotation_folder, 'hamer_hand_object_contacts.json'), 'w') as f:
        json.dump(all_segments_info, f)

if __name__ == "__main__":
    main()