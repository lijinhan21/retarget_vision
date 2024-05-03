import cv2
import os
import torch
import math
import json
import numpy as np

import plotly.graph_objects as go
import multiprocessing as mp

from enum import Enum
from collections import Counter
from scipy.optimize import linear_sum_assignment

from orion.utils.misc_utils import (
    get_video_seq_from_annotation,
    load_first_frame_from_hdf5_dataset, 
    load_image_from_hdf5_dataset, 
    overlay_xmem_mask_on_image, 
    depth_to_rgb, 
    plotly_draw_seg_image, 
    plotly_draw_image, 
    plotly_draw_3d_pcd, 
    plotly_draw_image_correspondences, 
    plotly_draw_image_with_object_keypoints, 
    resize_image_to_same_shape, 
    get_tracked_points_annotation,
    get_smplh_traj_annotation, 
    get_optical_flow_annotation,
    get_first_frame_annotation, 
    get_annotation_info,
    get_hamer_result,
    get_depth_seq_from_human_demo, 
    get_image_seq_from_human_demo)

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
from orion.utils.traj_utils import SimpleTrajProcessor, Interpolator
from orion.utils.correspondence_utils import CorrespondenceModel, find_most_repeated_number
from orion.utils.robosuite_utils import *
from orion.utils.log_utils import get_orion_logger
from orion.algos.retargeter_wrapper import Retargeter
from orion.algos.grasp_primitives import GraspPrimitive

ORION_LOGGER = get_orion_logger("orion")

class HandType(Enum):
    LEFT = 0
    RIGHT = 1

class Hand:
    def __init__(self, lr):
        self.lr = HandType.LEFT if lr == 'left' else HandType.RIGHT
        self.hand_object_contacts = []
        self.grasp_type = None

class SMPLHTraj:
    def __init__(self, ratio=1.0):
        self.smplh_traj = []
        self.video_smplh_ratio = ratio # num_frames_in_video / num_frames_in_smplh_traj
    
    def cal_idx_from_smplh_to_video(self, idx):
        return math.floor(idx * self.video_smplh_ratio)

    def cal_idx_from_video_to_smplh(self, idx):
        return math.floor(idx / self.video_smplh_ratio)
    
    def add_smplh_traj(self, smplh_traj):
        self.smplh_traj = smplh_traj

class HumanNode:
    def __init__(self, ratio=1.0):
        self.left_hand = Hand('left')
        self.right_hand = Hand('right')
        self.important_body_parts = []
        self.smplh_traj = SMPLHTraj(ratio)

    def add_smplh_traj(self, smplh_traj):
        self.smplh_traj.add_smplh_traj(smplh_traj)

class ObjectNode:
    def __init__(self, object_id=-1):
        self.object_id = object_id
        self.point_ids = []
        self.pcd_points = np.array([])
        self.pcd_colors = np.array([])

class PointNode:
    def __init__(self, point_id=-1, object_id=-1):
        self.point_id = point_id
        self.object_id = object_id

        self.pixel_point = np.array([-1, -1])
        self.world_point = np.array([-1, -1, -1])
        self.tracked_pixel_traj = np.array([])
        self.tracked_world_traj = np.array([])
        self.tracked_visibility_traj = np.array([])

class HandObjectInteractionGraph:
    def __init__(self):
        self.human_node = None
        self.object_nodes = []
        self.point_nodes = []
        self.object_contact_states = []
        
        self.segment_start_idx = -1
        self.segment_end_idx = -1
        self.human_video_annotation_path = ""

        self.retargeted_ik_traj = np.array([])
        self.grasp_type = [None, None]

    @property
    def segment_length(self):
        return self.segment_end_idx - self.segment_start_idx + 1

    def create_from_human_video(self, 
                                human_video_annotation_path, 
                                segment_start_idx, 
                                segment_end_idx, 
                                video_smplh_ratio=1.0,
                                use_smplh=True):
        self.segment_start_idx = segment_start_idx
        self.segment_end_idx = segment_end_idx
        self.human_video_annotation_path = human_video_annotation_path

        is_first_frame = (segment_start_idx == 0)
        if is_first_frame:
            _, human_annotation = get_first_frame_annotation(human_video_annotation_path)
        else:
            mask_file = f"{human_video_annotation_path}/masks.npz"
            if not os.path.exists(mask_file):
                raise ValueError(f"Mask file {mask_file} does not exist. You need to run XMem annotation first in order to proceed.")
            masks = np.load(mask_file)['arr_0']
            human_annotation = masks[segment_start_idx]

        tap_results = get_tracked_points_annotation(human_video_annotation_path)
        pred_tracks, pred_visibility = tap_results["pred_tracks"], tap_results["pred_visibility"]

        total_points = pred_tracks.shape[2]
        num_points_per_object = total_points // human_annotation.max()
        print("number of objects=", human_annotation.max())

        sampled_points = {}
        tracked_trajs = {}
        visibility_trajs = {}
        for object_id in range(1, human_annotation.max()+1):
            points_per_object = pred_tracks[0, segment_start_idx, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2]
            sampled_points[object_id] = points_per_object.detach().cpu().numpy()
            tracked_trajs[object_id] = pred_tracks[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2].detach().cpu().permute(1, 0, 2).numpy()
            visibility_trajs[object_id] = pred_visibility[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object].detach().cpu().permute(1, 0).numpy()

        # create object nodes and point nodes
        for object_id in range(1, human_annotation.max()+1):
            object_node = ObjectNode(object_id)
            object_node.point_ids = list(range((object_id - 1) * num_points_per_object, object_id * num_points_per_object))
            
            point_idx = (object_id - 1) * num_points_per_object
            for point in sampled_points[object_id]:
                point_node = PointNode()
                point_node.point_id = point_idx
                point_node.pixel_point = point
                point_node.object_id = object_id
                self.point_nodes.append(point_node)
                point_idx += 1

        # TODO: add object point clouds

        # create human node
        self.human_node = HumanNode(video_smplh_ratio)
        if use_smplh:
            # load whole smplh traj, and parse out the needed part
            smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)
            smplh_start_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_start_idx)
            smplh_end_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_end_idx + 1)
            self.human_node.add_smplh_traj(smplh_traj[smplh_start_idx:smplh_end_idx])
            print("video_idx=", segment_start_idx, segment_end_idx)
            print("smplh_idx=", smplh_start_idx, smplh_end_idx)

    def get_representative_images(self, num_images=5):
        img_idx = np.linspace(self.segment_start_idx, self.segment_end_idx, num_images, dtype=int)
        video_seq = get_video_seq_from_annotation(self.human_video_annotation_path)
        img_lst = [video_seq[idx] for idx in img_idx]
        return img_lst

    def get_retargeted_ik_traj(self, retargeter, offset={"link_RArm7": [0, 0, 0]}, num_waypoints=0, interpolation_steps=-1, interpolation_type='linear'):
        smplh_traj = self.human_node.smplh_traj.smplh_traj
        num_frames = len(smplh_traj)
        
        num_key_steps = num_waypoints + 2
        key_smplh_idx = np.linspace(0, num_frames-1, num_key_steps, dtype=int)
        key_smplh_traj = [smplh_traj[idx] for idx in key_smplh_idx]

        if interpolation_steps == -1:
            interpolation_steps = min(int(num_frames // num_key_steps) * 3, 40)
        interpolator = Interpolator(interpolation_type)
        
        key_retargeted_traj = []
        for i in range(num_key_steps):
            retargeted_traj, _, __ = retargeter.retarget(key_smplh_traj[i], offset=offset)
            key_retargeted_traj.append(retargeted_traj.copy())
        key_retargeted_traj = np.array(key_retargeted_traj)

        res = []
        for i in range(num_key_steps - 1):
            for j in range(interpolation_steps):
                res.append(interpolator(key_retargeted_traj[i], key_retargeted_traj[i+1], j, interpolation_steps))

        self.retargeted_ik_traj = np.array(res)
        return self.retargeted_ik_traj
    
    def get_retargeted_ik_traj_with_grasp_primitive(self,
                                                    retargeter,
                                                    offset={"link_RArm7": [0, 0, 0]},
                                                    num_waypoints=0,
                                                    interpolation_steps=-1,
                                                    interpolation_type='linear'):
        smplh_traj = self.human_node.smplh_traj.smplh_traj
        num_frames = len(smplh_traj)
        
        num_key_steps = num_waypoints + 2
        key_smplh_idx = np.linspace(0, num_frames-1, num_key_steps, dtype=int)
        key_smplh_traj = [smplh_traj[idx] for idx in key_smplh_idx]

        if interpolation_steps == -1:
            interpolation_steps = min(int(num_frames // num_key_steps) * 3, 40)
        interpolator = Interpolator(interpolation_type)
        
        key_retargeted_traj = []
        for i in range(num_key_steps):
            retargeted_traj, _, __ = retargeter.retarget(key_smplh_traj[i], offset=offset)
            key_retargeted_traj.append(retargeted_traj.copy())
        key_retargeted_traj = np.array(key_retargeted_traj)

        res = []
        for i in range(num_key_steps - 1):
            for j in range(interpolation_steps):
                res.append(interpolator(key_retargeted_traj[i], key_retargeted_traj[i+1], j, interpolation_steps))
        res = np.array(res)

        # calculate grasp primitive
        grasp_dict = GraspPrimitive()
        actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
        hand_primitive_l = grasp_dict.sequence_map_to_primitive(res[:, 13 + actuator_idxs])
        hand_primitive_r = grasp_dict.sequence_map_to_primitive(res[:, 32 + actuator_idxs])
        res[:, 13 + actuator_idxs] = np.tile(hand_primitive_l[0], (res.shape[0], 1))
        res[:, 32 + actuator_idxs] = np.tile(hand_primitive_r[0], (res.shape[0], 1))
        print("primitive for left right hand is", hand_primitive_l[1], hand_primitive_r[1])

        self.retargeted_ik_traj = res
        return self.retargeted_ik_traj