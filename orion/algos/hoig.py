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
    get_hand_object_contacts_annotation,
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

        self.smplh_traj = np.array([])

        self.retargeted_ik_traj = np.array([])
        self.grasp_type = [None, None]
        self.moving_arm = None

        self.camera_extrinsics = np.eye(4)
        self.camera_intrinsics = np.array([
            [909.83630371,   0.        , 651.97015381],
            [  0.        , 909.12280273, 376.37097168],
            [  0.        ,   0.        ,   1.        ],
        ])

    def get_pixel_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        pixel_trajs = []
        for point_node in selected_point_nodes:
            pixel_trajs.append(point_node.tracked_pixel_traj)
        return np.stack(pixel_trajs, axis=0)
    
    def get_visibility_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        visibility = []
        for point_node in selected_point_nodes:
            visibility.append(point_node.tracked_visibility_traj)
        return np.stack(visibility, axis=0)

    def get_point_list(self, point_nodes=None, mode="pixel"):
        if point_nodes is None:
            # use all point nodes if not specified
            point_nodes = self.point_nodes
        point_list = []
        for point_node in point_nodes:
            if mode == "pixel":
                point_list.append(point_node.pixel_point)
            elif mode == "world":
                point_list.append(point_node.world_point)
        return point_list

    def select_point_nodes(self, object_ids=[]):
        selected_point_nodes = []
        for point_node in self.point_nodes:
            if point_node.object_id in object_ids:
                selected_point_nodes.append(point_node)
        return selected_point_nodes
    
    def get_points_by_objects(self, object_ids=[], mode="pixel"):
        point_nodes = self.select_point_nodes(object_ids=object_ids)
        return self.get_point_list(point_nodes=point_nodes, mode=mode)

    def set_pixel_trajs(self, pixel_trajs):
        assert(pixel_trajs.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_pixel_traj = pixel_trajs[i]

    def set_visibility_trajs(self, visibility_traj):
        assert(visibility_traj.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_visibility_traj = visibility_traj[i]
            point_node.tracked_visibility = visibility_traj[i][0]

    @property
    def segment_length(self):
        return self.segment_end_idx - self.segment_start_idx + 1

    def create_from_human_video(self, 
                                human_video_annotation_path, 
                                segment_start_idx, 
                                segment_end_idx, 
                                segment_idx, 
                                retargeter,
                                grasp_dict_l, 
                                grasp_dict_r,
                                calibrate_grasp,
                                zero_pose_name,
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

        # Add trajectory for point nodes
        pixel_trajs = tap_results["pred_tracks"].squeeze().permute(1, 0, 2).detach().cpu().numpy()
        # get the visibility of point trajectories
        visibility = tap_results["pred_visibility"].squeeze().permute(1, 0).detach().cpu().numpy()

        pixel_trajs = pixel_trajs[:, segment_start_idx:segment_end_idx]
        visibility = visibility[:, segment_start_idx:segment_end_idx]
        self.set_pixel_trajs(pixel_trajs)
        self.set_visibility_trajs(visibility)

        depth_seg_seq = get_depth_seq_from_human_demo(human_demo_dataset_name, start_idx=segment_start_idx, end_idx=segment_end_idx)
        world_trajs = self.compute_world_trajs(pixel_trajs, depth_seg_seq, self.camera_intrinsics, self.camera_extrinsics)


        print("set points trajectory ok!")

        # TODO: add object point clouds

        # create human node
        self.human_node = HumanNode(video_smplh_ratio)
        if use_smplh:
            # load whole smplh traj, and parse out the needed part
            smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)
            smplh_start_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_start_idx)
            smplh_end_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_end_idx + 1)
            self.human_node.add_smplh_traj(smplh_traj[smplh_start_idx:smplh_end_idx])

            self.smplh_traj = self.human_node.smplh_traj.smplh_traj

            print("video_idx=", segment_start_idx, segment_end_idx)
            print("smplh_idx=", smplh_start_idx, smplh_end_idx)

            type_l = 'none'
            type_r = 'none'
            hand_object_contacts = get_hand_object_contacts_annotation(human_video_annotation_path)[segment_idx]
            type_l = 'close' if (hand_object_contacts['left']['contact_type'] == 'portable') else 'open'
            type_r = 'close' if (hand_object_contacts['right']['contact_type'] == 'portable') else 'open'
            self.hand_type = [type_l, type_r]

            # calculate grasp type
            self.get_grasp_type(grasp_dict_l, grasp_dict_r, type_l, type_r, calibrate_grasp, zero_pose_name, retargeter)

            # identify main moving arm in this step
            self.identify_moving_arm(retargeter)
    
    def identify_moving_arm(self, retargeter):
        L_targets = []
        R_targets = []
        for i in range(len(self.smplh_traj)):
            targets = retargeter.get_targets_from_smplh(self.smplh_traj[i])
            L_targets.append(targets['link_LArm7'][:3, 3])
            R_targets.append(targets['link_RArm7'][:3, 3])

        L_targets = np.array(L_targets)
        R_targets = np.array(R_targets)

        L_diff = np.linalg.norm(L_targets[1:] - L_targets[:-1], axis=1)
        R_diff = np.linalg.norm(R_targets[1:] - R_targets[:-1], axis=1)

        L_diff = np.mean(L_diff)
        R_diff = np.mean(R_diff)

        print("L_diff=", L_diff, "R_diff=", R_diff)

        cur_len = len(self.smplh_traj)
        while abs(L_diff - R_diff) < 0.006:
            # most likely both arms are moving. Then we need to identify the arm is moves in the latter part of the trajectory
            cur_len = int(cur_len * 0.5)
            L_diff = np.linalg.norm(L_targets[-(cur_len - 1):] - L_targets[-cur_len: -1], axis=1)
            R_diff = np.linalg.norm(R_targets[-(cur_len - 1):] - R_targets[-cur_len: -1], axis=1)

            L_diff = np.mean(L_diff)
            R_diff = np.mean(R_diff)

            print("cur_len=", cur_len, "L_diff=", L_diff, "R_diff=", R_diff)
        
        if L_diff > R_diff:
            self.moving_arm = 'L'
        else:
            self.moving_arm = 'R'

        print("moving arm is", self.moving_arm)

    def get_grasp_type(self, grasp_dict_l, grasp_dict_r, type_l, type_r, calibrate_grasp, zero_pose_name, retargeter):
        res = []

        actuator_idxs = np.array([0, 1, 8, 10, 4, 6])

        for i in range(len(self.smplh_traj)):
            # retargeted_traj, _, __ = retargeter.retarget(self.smplh_traj[i])
            retargeted_traj = retargeter.retarget(self.smplh_traj[i])
            res.append(retargeted_traj)
        res = np.array(res)
        
        
        if calibrate_grasp:
            hand_primitive_l = (grasp_dict_l.get_joint_angles(zero_pose_name), zero_pose_name)
            hand_primitive_r = (grasp_dict_r.get_joint_angles(zero_pose_name), zero_pose_name)
        else:
            hand_primitive_l = grasp_dict_l.sequence_map_to_primitive(res[:, 13 + actuator_idxs], type=type_l)
            hand_primitive_r = grasp_dict_r.sequence_map_to_primitive(res[:, 32 + actuator_idxs], type=type_r)
        
        self.grasp_type = [hand_primitive_l[1], hand_primitive_r[1]]
        print("grasp type=", self.grasp_type)

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
        print("key smplh idx", key_smplh_idx)

        # calculated retargeted ik traj for the whole trajectory, to ensure the results of ik is reasonable
        all_retargeted_traj = []
        for i in range(num_frames):
            retargeted_traj, _, __ = retargeter.retarget(smplh_traj[i], offset=offset)
            all_retargeted_traj.append(retargeted_traj.copy())
        # extract key retargeted traj using key idx
        key_retargeted_traj = np.array([all_retargeted_traj[idx] for idx in key_smplh_idx])

        # interpolate between key retargeted traj
        if interpolation_steps == -1:
            interpolation_steps = min(int(num_frames // num_key_steps) * 3, 30)
        interpolator = Interpolator(interpolation_type)
        print("interpolation steps", interpolation_steps)

        res = []
        for i in range(num_key_steps - 1):
            for j in range(interpolation_steps):
                res.append(interpolator(key_retargeted_traj[i], key_retargeted_traj[i+1], j, interpolation_steps))

        self.retargeted_ik_traj = np.array(res)
        return self.retargeted_ik_traj

    def get_retargeted_ik_traj_with_grasp_primitive(self,
                                                    retargeter,
                                                    grasp_dict_l,
                                                    grasp_dict_r,
                                                    calibrate_grasp,
                                                    zero_pose_name,
                                                    offset={"link_RArm7": [0, 0, 0]},
                                                    num_waypoints=0,
                                                    interpolation_steps=-1,
                                                    interpolation_type='linear'):
        smplh_traj = self.human_node.smplh_traj.smplh_traj
        num_frames = len(smplh_traj)
        
        num_key_steps = num_waypoints + 2
        key_smplh_idx = np.linspace(0, num_frames-1, num_key_steps, dtype=int)
        print("key smplh idx", key_smplh_idx)

        # calculated retargeted ik traj for the whole trajectory, to ensure the results of ik is reasonable
        all_retargeted_traj = []
        for i in range(num_frames):
            retargeted_traj, _, __ = retargeter.retarget(smplh_traj[i], offset=offset)
            all_retargeted_traj.append(retargeted_traj.copy())
        # extract key retargeted traj using key idx
        key_retargeted_traj = np.array([all_retargeted_traj[idx] for idx in key_smplh_idx])

        # interpolate between key retargeted traj
        if interpolation_steps == -1:
            interpolation_steps = min(int(num_frames // num_key_steps) * 3, 30)
        interpolator = Interpolator(interpolation_type)
        print("interpolation steps", interpolation_steps)

        res = []
        for i in range(num_key_steps - 1):
            for j in range(interpolation_steps):
                res.append(interpolator(key_retargeted_traj[i], key_retargeted_traj[i+1], j, interpolation_steps))
        res = np.array(res)

        # calculate grasp primitive
        actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
        # if calibrate_grasp:
        #     grasp_dict_l.calibrate(res[0, 13 + actuator_idxs], zero_pose_name)
        #     grasp_dict_r.calibrate(res[0, 32 + actuator_idxs], zero_pose_name)
        # hand_primitive_l = grasp_dict_l.sequence_map_to_primitive(res[:, 13 + actuator_idxs])
        # hand_primitive_r = grasp_dict_r.sequence_map_to_primitive(res[:, 32 + actuator_idxs])
        
        # Hack: let first frame hand pose be zero pose; other frame use nn to determine
        if calibrate_grasp:
            # calibrate (set offset). can be commented!
            # grasp_dict_l.calibrate(res[0, 13 + actuator_idxs], zero_pose_name)
            # grasp_dict_r.calibrate(res[0, 32 + actuator_idxs], zero_pose_name)

            hand_primitive_l = (grasp_dict_l.get_joint_angles(zero_pose_name), zero_pose_name)
            hand_primitive_r = (grasp_dict_r.get_joint_angles(zero_pose_name), zero_pose_name)
        else:
            hand_primitive_l = grasp_dict_l.sequence_map_to_primitive(res[:, 13 + actuator_idxs])
            hand_primitive_r = grasp_dict_r.sequence_map_to_primitive(res[:, 32 + actuator_idxs])
        
        res[:, 13 + actuator_idxs] = np.tile(hand_primitive_l[0], (res.shape[0], 1))
        res[:, 32 + actuator_idxs] = np.tile(hand_primitive_r[0], (res.shape[0], 1))
        print("primitive for left right hand is", hand_primitive_l[1], hand_primitive_r[1])
        self.grasp_type = [hand_primitive_l[1], hand_primitive_r[1]]

        self.retargeted_ik_traj = res
        return self.retargeted_ik_traj