import cv2
import os
import torch
import math
import numpy as np

import plotly.graph_objects as go
import multiprocessing as mp

from enum import Enum
from collections import Counter
from scipy.optimize import linear_sum_assignment

from orion.utils import optim_utils
from orion.utils.misc_utils import (
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
from orion.utils.traj_utils import SimpleTrajProcessor
from orion.utils.correspondence_utils import CorrespondenceModel, find_most_repeated_number

from orion.utils.misc_utils import (
    create_point_clouds_from_keypoints, 
    simple_filter_outliers, 
    transform_points)

from orion.utils.hand_utils import (
    compute_thumb_index_joints,
    get_finger_joint_points,
    get_thumb_tip_points,
    get_thumb_dip_points,
    get_index_tip_points,
    get_index_dip_points,
    SimpleEEFAction,
    InteractionAffordance,
)

from orion.utils.log_utils import get_orion_logger

ORION_LOGGER = get_orion_logger("orion")

class OOGMode(Enum):
    NULL = 0
    FREE_MOTION = 1
    MOTION_GRIPPER_CLOSE = 2
    MOTION_GRIPPER_OPEN = 3

class Node():
    def __init__(self):
        pass

class SimpleEEFNode(Node):
    def __init__(self, name="hand"):
        super().__init__()
        assert(name in ["hand", "gripper"]), "name must be either `hand` or `gripper`"
        self.name = name
        self.thum_index_distances = []
        self.action = SimpleEEFAction.NULL
        self.interaction_affordance = InteractionAffordance()

    def set_eef_action(self, action):
        assert(action in SimpleEEFAction), "action must be a SimpleEEFAction"
        self.action = action

    def get_eef_action(self, verbose=False):
        if verbose:
            ORION_LOGGER.debug("Current eef action is: {}".format(self.action))
        return self.action

    def set_thumb_index_distances(self, distances):
        self.thum_index_distances = distances

    def visualize_thumb_index_dist_curve(self, threshold=None):
        """Visualize the thumb index distance curve"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.thum_index_distances, mode="lines"))
        if threshold is None:
            threshold = (max(self.thum_index_distances) + min(self.thum_index_distances)) / 2
            
        fig.add_trace(go.Scatter(y=[threshold]*len(self.thum_index_distances), mode="lines", name="threshold"))
        fig.show()

    @classmethod
    def from_dict(self, eef_dict):
        """Create the EEFNode class from saved dictionary"""
        eef_node = SimpleEEFNode(name=eef_dict["name"])
        eef_node.thum_index_distances = eef_dict["thum_index_distances"]
        eef_node.action = eef_dict["action"]
        eef_node.interaction_affordance = InteractionAffordance.from_dict(eef_dict["interaction_affordance"])
        return eef_node
    
    def to_dict(self):
        return {
            "name": self.name,
            "thum_index_distances": self.thum_index_distances,
            "action": self.action,
            "interaction_affordance": self.interaction_affordance.to_dict(),
        }

class ObjectNode(Node):
    def __init__(self, object_id=-1):
        super().__init__()
        self.object_id = object_id
        self.points = []
        self.pcd_points = np.array([])
        self.pcd_colors = np.array([])

    def add_point(self, point_node):
        self.points.append(point_node.idx)

    def get_point(self, idx):
        return self.points[idx] - 1

    def __repr__(self):
        return f"ObjectNode: {self.object_id} with {len(self.points)} points"
    
    def to_dict(self):
        return {
            "object_id": self.object_id,
            "points": self.points,
            "pcd_points": self.pcd_points,
            "pcd_colors": self.pcd_colors,
        }
    
    @classmethod
    def from_dict(self, object_dict):
        """Create the ObjectNode class from saved dictionary"""
        object_node = ObjectNode(object_id=object_dict["object_id"])
        object_node.points = object_dict["points"]
        object_node.pcd_points = object_dict["pcd_points"]
        object_node.pcd_colors = object_dict["pcd_colors"]
        return object_node

class PointNode(Node):
    def __init__(self, idx=-1, object_id=-1):
        super().__init__()
        self.idx = idx
        self.object_id = object_id
        self.pixel_point = np.array([-1, -1])
        self.world_point = np.array([-1, -1, -1])
        self.tracked_pixel_traj = np.array([])
        self.tracked_world_traj = np.array([])
        self.invalid_world_traj = False

        self.tracked_visibility = True
        self.tracked_visibility_traj = np.array([])

        self.baseline_optical_flow_pixel_traj = np.array([])
        self.baseline_optical_flow_world_traj = np.array([])

        self.baseline_thumb_world_traj = np.array([])
        self.baseline_index_world_traj = np.array([])

        self.internalized_motion_weights = None

    def to_dict(self):
        return {
            "idx": self.idx,
            "object_id": self.object_id,
            "pixel_point": self.pixel_point,
            "world_point": self.world_point,
            "tracked_pixel_traj": self.tracked_pixel_traj,
            "tracked_world_traj": self.tracked_world_traj,
            "tracked_visibility": self.tracked_visibility,
            "tracked_visibility_traj": self.tracked_visibility_traj,
            "invalid_world_traj": self.invalid_world_traj,

            # "baseline_optical_flow_pixel_traj": self.baseline_optical_flow_pixel_traj,
            # "baseline_optical_flow_world_traj": self.baseline_optical_flow_world_traj,

            # "baseline_thumb_world_traj": self.baseline_thumb_world_traj,
            # "baseline_index_world_traj": self.baseline_index_world_traj,
        }

    @classmethod
    def from_dict(self, point_dict):
        """Create the PointNode class from saved dictionary"""
        point_node = PointNode(idx=point_dict["idx"], object_id=point_dict["object_id"])
        point_node.pixel_point = point_dict["pixel_point"]
        point_node.world_point = point_dict["world_point"]
        point_node.tracked_pixel_traj = point_dict["tracked_pixel_traj"]
        point_node.tracked_world_traj = point_dict["tracked_world_traj"]
        point_node.tracked_visibility = point_dict["tracked_visibility"]
        point_node.tracked_visibility_traj = point_dict["tracked_visibility_traj"]
        point_node.invalid_world_traj = point_dict["invalid_world_traj"]

        # point_node.baseline_optical_flow_pixel_traj = point_dict["baseline_optical_flow_pixel_traj"]
        # point_node.baseline_optical_flow_world_traj = point_dict["baseline_optical_flow_world_traj"]

        # point_node.baseline_thumb_world_traj = point_dict["baseline_thumb_world_traj"]
        # point_node.baseline_index_world_traj = point_dict["baseline_index_world_traj"]
        return point_node

    def __repr__(self) -> str:
        return f"PointNode: {self.idx} with object_id {self.object_id} at pixel {self.pixel_point} and world {self.world_point}. Its stored trajectory has a length of {len(self.tracked_pixel_traj)}."

class OpenWorldObjectSceneGraph():
    def __init__(self, 
                 task_id=-1, 
                 eef_name="hand", 
                 name="placeholder",
                 debug_mode=False):

        self.task_id = task_id
        self.input_image = None
        self.input_annotation = None
        self.input_depth = None
        self.points = None
        self.object_nodes = []
        self.point_nodes = []
        self.points_affordance = None # this variable store the weights of actions
        self.is_first_frame = False

        self.eef_node = SimpleEEFNode(name=eef_name)

        self.name = name

        self.camera_extrinsics = None
        self.camera_intrinsics = None

        self.total_points = 0

        self.debug_mode = debug_mode

        self.oog_mode = OOGMode.NULL

        self.manipulate_object_id = -1
        self.reference_object_id = -1

        self.segment_start_idx = -1
        self.segment_end_idx = -1

        self.contact_states = []

    def set_oog_mode(self, oog_mode):
        assert(oog_mode in OOGMode), "oog_mode must be a OOGMode"
        self.oog_mode = oog_mode

    def get_oog_mode(self):
        return self.oog_mode
    
    def set_manipulate_object_id(self, id):
        self.manipulate_object_id = id

    def get_manipulate_object_id(self):
        return self.manipulate_object_id
    
    def set_reference_object_id(self, id):
        self.reference_object_id = id

    def get_reference_object_id(self):
        return self.reference_object_id
    
    def decide_gripper_action(self):
        if self.eef_node.action == SimpleEEFAction.OPEN:
            self.set_oog_mode(OOGMode.MOTION_GRIPPER_OPEN)
        elif self.eef_node.action == SimpleEEFAction.CLOSE:
            self.set_oog_mode(OOGMode.MOTION_GRIPPER_CLOSE)

    def to_dict(self):
        save_data = {}
        save_data["task_id"] = self.task_id
        save_data["input_image"] = self.input_image
        save_data["input_annotation"] = self.input_annotation
        save_data["input_depth"] = self.input_depth
        save_data["object_nodes"] = [object_node.to_dict() for object_node in self.object_nodes]
        save_data["point_nodes"] = [point_node.to_dict() for point_node in self.point_nodes]
        save_data["eef_node"] = self.eef_node.to_dict()
        save_data["eef_node"] = self.eef_node.to_dict()
        save_data["camera_extrinsics"] = self.camera_extrinsics
        save_data["camera_intrinsics"] = self.camera_intrinsics
        save_data["total_points"] = self.total_points
        save_data["manipulate_object_id"] = self.manipulate_object_id
        save_data["reference_object_id"] = self.reference_object_id
        save_data["oog_mode"] = self.oog_mode
        save_data["segment_start_idx"] = self.segment_start_idx
        save_data["segment_end_idx"] = self.segment_end_idx
        save_data["contact_states"] = self.contact_states
        return save_data
    
    def from_dict(self, load_data):
        self.task_id = load_data["task_id"]
        self.input_image = load_data["input_image"]
        self.input_annotation = load_data["input_annotation"]
        self.input_depth = load_data["input_depth"]
        self.object_nodes = [ObjectNode.from_dict(object_dict) for object_dict in load_data["object_nodes"]]
        self.point_nodes = [PointNode.from_dict(point_dict) for point_dict in load_data["point_nodes"]]
        self.eef_node = SimpleEEFNode.from_dict(load_data["eef_node"])
        self.camera_extrinsics = load_data["camera_extrinsics"]
        self.camera_intrinsics = load_data["camera_intrinsics"]
        self.total_points = load_data["total_points"]

        self.manipulate_object_id = load_data["manipulate_object_id"]
        self.reference_object_id = load_data["reference_object_id"]
        self.oog_mode = load_data["oog_mode"]

        self.segment_start_idx = load_data["segment_start_idx"]
        self.segment_end_idx = load_data["segment_end_idx"]
        try:
            self.contact_states = load_data["contact_states"]
        except:
            self.contact_states = []


    def save_to_ckpt(self, ckpt_path):
        """Save scene graph to checkpoints

        Args:
            ckpt_path (str): a path to the checkpoint file that will store the information needed for scene graph.
        """
        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        save_data = self.to_dict()
        torch.save(save_data, ckpt_path)

    def load_from_ckpt(self, ckpt_path):
        """Load scene graph from checkpoints

        Args:
            ckpt_path (str): a path to the checkpoint file that stored the information needed for scene graph.
        """
        load_data = torch.load(ckpt_path)
        self.from_dict(load_data)

    def compute_contact_states(self, dist_threshold=0.01, intersection_threshold=100):
        pcd_list = []
        for object_id in self.object_ids:
            pcd_array, _ = self.get_objects_3d_points(object_id=object_id,
                                                                        filter=False)
            pcd = create_o3d_from_points_and_color(pcd_array)
            pcd_list.append(pcd)

        dists = []
        contact_states = []

        max_intersection = 0

        dists = []
        contact_states = []

        for i in range(len(pcd_list)):
            for j in range(i+1, len(pcd_list)):
                dists = pcd_list[i].compute_point_cloud_distance(pcd_list[j])
                intersections = [d for d in dists if d < dist_threshold]
                if len(intersections) > intersection_threshold:
                    contact_states.append((i+1, j+1))
        self.contact_states = contact_states


    def create_object_nodes_from_annotation(self, annotation):
        self.object_nodes = []
        for object_id in np.unique(annotation):
            if object_id > 0:
                object_node = ObjectNode(object_id=object_id)
                self.object_nodes.append(object_node)

    def create_point_nodes_from_keypoints(self, points_dict):
        for object_node in self.object_nodes:
            object_id = object_node.object_id
            for point in points_dict[object_id]:
                self.total_points += 1
                point_node = PointNode(idx=self.total_points, object_id=object_id)
                point_node.pixel_point = point
                self.point_nodes.append(point_node)
                object_node.add_point(point_node)

    def generate_from_image_and_points(self, 
                                 input_image, 
                                 input_annotation,
                                 points_dict,
                                 is_first_frame):
        self.input_image = np.ascontiguousarray(input_image)
        self.input_annotation = input_annotation
        assert(self.input_image.shape[:2] == self.input_annotation.shape[:2]), "Input image and annotation must have the same size"
        self.create_object_nodes_from_annotation(self.input_annotation)
        assert(max(points_dict.keys()) == self.input_annotation.max())
        self.create_point_nodes_from_keypoints(points_dict)
        self.is_first_frame = is_first_frame

    def generate_from_human_demo(self, 
                                 human_video_annotation_path,
                                 segment_start_idx=0,
                                 segment_end_idx=None,
                                 mode="lab",
                                 previous_graph=None,
                                 filter_annotation=False,
                                 plane_estimation_depth_trunc=5.0,
                                 plane_estimation_kwargs={
                                        "ransac_n": 3,
                                        "num_iterations": 1000,
                                        "distance_threshold": 0.01
                                 },
                                 baseline_optical_flow=False,
                                 ):
        """Generate OOG from human video input. If (segment_start_idx, segment_end_idx) == (0, None), it means that no segmentation is applied and the whole video is considered. 

        Args:
            human_video_annotation_path (_type_): _description_
            segment_start_idx (int, optional): _description_. Defaults to 0.
            segment_end_idx (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to "lab".
        """
        assert(mode in ["lab", "iphone"]), "Mode must be either `lab` or `iphone`"
        if mode == "lab":
            z_up = False
        elif mode == "iphone":
            z_up = True
        else:
            raise NotImplementedError
        
        self.segment_start_idx = segment_start_idx
        self.segment_end_idx = segment_end_idx

        is_first_frame = False
        if previous_graph is None:
            is_first_frame = True

        # TODO: Take care of general cases of following images
        # 1.1 load segmentation and the image
        if is_first_frame:
            _, human_annotation = get_first_frame_annotation(human_video_annotation_path)
        else:
            mask_file = f"{human_video_annotation_path}/masks.npz"
            if not os.path.exists(mask_file):
                raise ValueError(f"Mask file {mask_file} does not exist. You need to run XMem annotation first in order to proceed.")
            masks = np.load(mask_file)['arr_0']
            human_annotation = masks[segment_start_idx]

        # 1.2 load tap results from pre-annotation stage
        tap_results = get_tracked_points_annotation(human_video_annotation_path)
        # pred_tracks: (B, T, NUM_OBJECTS * NUM_POINTS, 2), where B=1
        pred_tracks, pred_visibility = tap_results["pred_tracks"], tap_results["pred_visibility"]

        if baseline_optical_flow:
            optical_flow_results = get_optical_flow_annotation(human_video_annotation_path)
            pred_tracks = torch.from_numpy(optical_flow_results).unsqueeze(0).permute(0, 2, 1, 3)
            pred_visibility = torch.ones_like(pred_tracks[:, :, :, 0])

            tap_results = {
                "pred_tracks": pred_tracks,
                "pred_visibility": pred_visibility
            }

        total_points = pred_tracks.shape[2]
        num_points_per_object = total_points // human_annotation.max()

        # TODO: needs to simplify this part
        sampled_points = {}
        tracked_trajs = {}
        visibility_trajs = {}
        for object_id in range(1, human_annotation.max()+1):
            points_per_object = pred_tracks[0, segment_start_idx, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2]
            sampled_points[object_id] = points_per_object.detach().cpu().numpy()
            tracked_trajs[object_id] = pred_tracks[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2].detach().cpu().permute(1, 0, 2).numpy()
            visibility_trajs[object_id] = pred_visibility[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object].detach().cpu().permute(1, 0).numpy()


        # 2.1 get the config info from the annotation file which specifies the stored demonstration data
        config_info = get_annotation_info(human_video_annotation_path)
        human_demo_dataset_name = config_info["original_file"]
        image_seg_seq = get_image_seq_from_human_demo(human_demo_dataset_name, start_idx=segment_start_idx, end_idx=segment_end_idx)
        depth_seg_seq = get_depth_seq_from_human_demo(human_demo_dataset_name, start_idx=segment_start_idx, end_idx=segment_end_idx)
        # initialize the images and the initial points on pixels.
        self.generate_from_image_and_points(
            input_image=image_seg_seq[0],
            input_annotation=human_annotation,
            points_dict=sampled_points,
            is_first_frame=is_first_frame
        )
        # Get the image and depth sequence given (segment_start_idx, segment_end_idx)

        # 2.2 load camera extrinsics, and intrinsics, and the depth image corresonding to 
        recon_info = load_reconstruction_info_from_human_demo(human_demo_dataset_name)
        self.set_camera_extrinsics(np.eye(4))
        self.set_camera_intrinsics(recon_info["intrinsics"])
        self.load_depth(depth_seg_seq[0])

        if self.is_first_frame:

            T_xy_plane_align, _ = self.estimate_plane_rotation(z_up=z_up,
                                                               depth_trunc=plane_estimation_depth_trunc,
                                                               plane_estimation_kwargs=plane_estimation_kwargs)
            self.set_camera_extrinsics(T_xy_plane_align)

            if self.debug_mode:
                ORION_LOGGER.debug(f"Estimated transformation matrix is: {T_xy_plane_align}")
        else:
            self.set_camera_extrinsics(previous_graph.camera_extrinsics)

        # filter annotation
        if filter_annotation:
            self.filter_annotation()

        # get tap trajectories on pixels
        pixel_trajs = tap_results["pred_tracks"].squeeze().permute(1, 0, 2).detach().cpu().numpy()
        # get the visibility of point trajectories
        visibility = tap_results["pred_visibility"].squeeze().permute(1, 0).detach().cpu().numpy()

        pixel_trajs = pixel_trajs[:, segment_start_idx:segment_end_idx]
        visibility = visibility[:, segment_start_idx:segment_end_idx]

        if self.debug_mode:
            ORION_LOGGER.debug(f"Pixel traj shape: {pixel_trajs.shape}")
        # compute tap trajectories in 3D space
        world_trajs = self.compute_world_trajs(pixel_trajs, depth_seg_seq, self.camera_intrinsics, self.camera_extrinsics)
        self.set_pixel_trajs(pixel_trajs)
        self.set_world_trajs(world_trajs)
        self.set_visibility_trajs(visibility)


        # create eef node of hand from hamer annotation
        hamer_result, _ = get_hamer_result(human_video_annotation_path)
        overall_distances = compute_thumb_index_joints(human_video_annotation_path)
        overall_distances[overall_distances==0] = min(overall_distances[overall_distances!=0])
        hand_threshold = (max(overall_distances) + min(overall_distances)) / 2
        segment_distances = overall_distances[segment_start_idx:segment_end_idx]
        vitpose_detections = hamer_result["vitpose_detections"][segment_start_idx][:, 0:2]

        self.eef_node.set_thumb_index_distances(segment_distances)
        eef_action = SimpleEEFAction.NULL
        if find_most_repeated_number(segment_distances > hand_threshold):
            eef_action = SimpleEEFAction.OPEN
        else:
            eef_action = SimpleEEFAction.CLOSE

        self.eef_node.set_eef_action(eef_action)
        thumb_tip_points = get_thumb_tip_points(vitpose_detections, depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics)
        index_tip_points = get_index_tip_points(vitpose_detections, depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics)

        thumb_tip_estimated_location = np.mean(thumb_tip_points, axis=0)
        index_tip_estimated_location = np.mean(index_tip_points, axis=0)
        interaction_centroid = (thumb_tip_estimated_location + index_tip_estimated_location) / 2

        self.eef_node.interaction_affordance.set_affordance_thumb_tip(thumb_tip_estimated_location)
        self.eef_node.interaction_affordance.set_affordance_index_tip(index_tip_estimated_location)
        self.eef_node.interaction_affordance.set_affordance_centroid(interaction_centroid)

        self.human_video_annotation_path = human_video_annotation_path
        self.human_demo_dataset_name = human_demo_dataset_name

        if baseline_optical_flow:
            optical_flow_3d_trajs = self.init_optical_flow_baseline_trajs()
            self.set_world_trajs(optical_flow_3d_trajs)

        return {
            "human_demo_dataset_name": human_demo_dataset_name,
        }
   
    def get_thumb_index_trajectories(self):
        hamer_result, _ = get_hamer_result(self.human_video_annotation_path)

        thumb_tip_trajs = []
        thumb_dip_trajs = []
        index_tip_trajs = []
        index_dip_trajs = []
        vitpose_detections = hamer_result["vitpose_detections"]
        depth_seg_seq = get_depth_seq_from_human_demo(self.human_demo_dataset_name)

        for i in range(self.segment_start_idx, self.segment_end_idx):
            thumb_tip_points = get_thumb_tip_points(vitpose_detections[i][:, 0:2], 
                                                    depth_seg_seq[i], 
                                                    self.camera_intrinsics, 
                                                    self.camera_extrinsics)
            thumb_tip_mean = np.mean(thumb_tip_points, axis=0)
            index_tip_points = get_index_tip_points(vitpose_detections[i][:, 0:2],
                                                    depth_seg_seq[i], 
                                                    self.camera_intrinsics, 
                                                    self.camera_extrinsics)
            index_tip_mean = np.mean(index_tip_points, axis=0)

            thumb_dip_points = get_thumb_dip_points(vitpose_detections[i][:, 0:2],
                                                    depth_seg_seq[i], 
                                                    self.camera_intrinsics, 
                                                    self.camera_extrinsics)
            thumb_dip_mean = np.mean(thumb_dip_points, axis=0)
            index_dip_points = get_index_dip_points(vitpose_detections[i][:, 0:2],
                                                    depth_seg_seq[i], 
                                                    self.camera_intrinsics, 
                                                    self.camera_extrinsics)
            index_dip_mean = np.mean(index_dip_points, axis=0)


            thumb_tip_trajs.append(thumb_tip_mean)
            index_tip_trajs.append(index_tip_mean)
            thumb_dip_trajs.append(thumb_dip_mean)
            index_dip_trajs.append(index_dip_mean)
        thumb_tip_trajs = np.array(thumb_tip_trajs)
        thumb_dip_trajs = np.array(thumb_dip_trajs)
        index_tip_trajs = np.array(index_tip_trajs)
        index_dip_trajs = np.array(index_dip_trajs)
        return thumb_tip_trajs, thumb_dip_trajs, index_tip_trajs, index_dip_trajs


    

    def generate_from_robot_demo(self,
                                 input_image,
                                 input_depth,
                                 reference_graph,
                                 is_first_frame,
                                 camera_intrinsics,
                                 camera_extrinsics,
                                 input_annotation=None,
                                 correspondence_model=None,
                                 verbose=True):
        if correspondence_model is None:
            correspondence_model = CorrespondenceModel()
        self.input_image = np.ascontiguousarray(input_image)
        if input_annotation is None:
            # create annotation first
            input_annotation = correspondence_model.segment_image(self.input_image)

        self.set_camera_extrinsics(camera_extrinsics)
        self.set_camera_intrinsics(camera_intrinsics)
        self.input_depth = input_depth

        new_current_annotation_mask, hungarian_matrix = correspondence_model.object_correspondence(
            current_obs_image_input=self.input_image,
            current_annotation_mask_input=input_annotation,
            ref_image_input=reference_graph.input_image,
            ref_annotation_mask_input=reference_graph.input_annotation,
            topk=20,             
        )
        new_current_annotation_mask = resize_image_to_same_shape(new_current_annotation_mask, self.input_image)

        hungarian_matched_annotation_mask = np.zeros_like(new_current_annotation_mask)

        row_ind, col_ind = linear_sum_assignment(hungarian_matrix, maximize=True)
        if verbose:
            ORION_LOGGER.debug(f"col ind: {col_ind}")
            ORION_LOGGER.debug("The unique set of indices are: {}".format(np.unique(new_current_annotation_mask)))
        for i in range(len(col_ind)):
            if verbose:
                ORION_LOGGER.info(f"{col_ind[i]}: {row_ind[i]}")
            hungarian_matched_annotation_mask[np.where(input_annotation == col_ind[i] + 1)] = row_ind[i] + 1
        self.input_annotation = hungarian_matched_annotation_mask
        for i in range(hungarian_matched_annotation_mask.max() + 1):
            ORION_LOGGER.debug("new mask: {} : {}".format(i, np.sum(hungarian_matched_annotation_mask == i)))

        # The annotation, however, might be wrong due to rough thresholding. We will use spatial correspondence to further correct the annotation

        # filter input annotation
        self.filter_annotation()
        query_points = reference_graph.get_point_list(mode="pixel")
        aff, feature_list, corresponding_points = correspondence_model.spatial_correspondence(
                        src_image=reference_graph.input_image,
                        tgt_image=self.input_image,
                        query_points=query_points,
                        src_annotation=reference_graph.input_annotation,
                        tgt_annotation=self.input_annotation,
                    )
        points_dict = {}

        for i, reference_point_node in enumerate(reference_graph.point_nodes):
            if reference_point_node.object_id not in points_dict:
                points_dict[reference_point_node.object_id] = []
            points_dict[reference_point_node.object_id].append(corresponding_points[i])

        new_mapping_votes = {}
        for object_id in points_dict.keys():
             new_mapping_votes[object_id] = []
             for point in points_dict[object_id]:
                if self.input_annotation[point[1], point[0]] != 0:
                    new_mapping_votes[object_id].append(self.input_annotation[point[1], point[0]])
        ORION_LOGGER.debug("Before filtering: {}".format(new_mapping_votes))
        
        new_annotation = np.zeros_like(self.input_annotation)

        initial_ids_matching = {}
        for object_id in new_mapping_votes:
            initial_ids_matching[object_id] = find_most_repeated_number(new_mapping_votes[object_id])
        
        for object_id_1 in initial_ids_matching.keys():
            for object_id_2 in  initial_ids_matching.keys():
                if object_id_1 != object_id_2:
                    if (initial_ids_matching[object_id_1] == initial_ids_matching[object_id_2]):
                        counter_1 = Counter(new_mapping_votes[object_id_1])
                        counter_2 = Counter(new_mapping_votes[object_id_2])
                        duplicate_element = initial_ids_matching[object_id_1]
                        if counter_1[duplicate_element] > counter_2[duplicate_element]:
                            new_mapping_votes[object_id_2] = [item for item in new_mapping_votes[object_id_2] if item != duplicate_element]
                        else:
                            new_mapping_votes[object_id_1] = [item for item in new_mapping_votes[object_id_1] if item != duplicate_element]
        ORION_LOGGER.debug("After filtering: {}".format(new_mapping_votes))
        for object_id in new_mapping_votes:
            original_id = find_most_repeated_number(new_mapping_votes[object_id])
            new_annotation[np.where(self.input_annotation==original_id)] = object_id

        self.input_annotation = new_annotation

        self.create_object_nodes_from_annotation(self.input_annotation)
        # assert(max(points_dict.keys()) == self.input_annotation.max())
        self.create_point_nodes_from_keypoints(points_dict)
        self.create_world_points()
        self.is_first_frame = is_first_frame
        return aff
   
    def filter_annotation(self, nb_neighbors=60, std_ratio=0.7):
        assert(self.input_depth is not None and self.input_annotation is not None and self.camera_intrinsics is not None), "Input depth, annotation, camera intrinsics, and camera extrinsics must be set"

        # we do this separately so that we can exclude points that are misplaced inside other segmentations
        for object_id in self.object_ids:
            segmented_depth = self.input_depth * (self.input_annotation == object_id)
            new_segmentation = filter_pcd(segmented_depth, self.camera_intrinsics, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            # find indices that new_segmentation and segmented_depth does not overlap
            self.input_annotation[self.input_annotation == object_id] = (new_segmentation * (self.input_annotation == object_id) * object_id)[self.input_annotation == object_id]

    def compute_world_trajs(self, 
                            tracked_pixel_trajs, 
                            depth_seq, 
                            camera_intrinsics_matrix, 
                            camera_extrinsics_matrix):
        points_list = []
        for t in range(tracked_pixel_trajs.shape[1]):
            points = create_point_clouds_from_keypoints(tracked_pixel_trajs[:, t], depth_seq[t], camera_intrinsics_matrix)
            points_list.append(points)
        points_list = np.stack(points_list, axis=0)
        world_trajs = []
        for i in range(tracked_pixel_trajs.shape[0]):
            segment = points_list[:, i]
            segment = simple_filter_outliers(segment)
            segment = transform_points(camera_extrinsics_matrix, segment)
            world_trajs.append(segment)
        return np.stack(world_trajs, axis=0)
    
    def init_optical_flow_baseline_trajs(self):
        trajs_file = os.path.join(self.human_video_annotation_path, 
                                  "dense_trajs.pt")
        trajs = torch.load(trajs_file)
        depth_seg_seq = get_depth_seq_from_human_demo(self.human_demo_dataset_name, 
                                                      start_idx=self.segment_start_idx, 
                                                      end_idx=self.segment_end_idx)

        optical_flow_3d_trajs = self.compute_world_trajs(trajs[:, self.segment_start_idx:self.segment_end_idx], 
                                                      depth_seg_seq, 
                                                      self.camera_intrinsics, 
                                                      self.camera_extrinsics)
        
        for point_idx in range(len(self.point_nodes)):
            self.point_nodes[point_idx].baseline_optical_flow_pixel_traj = trajs[point_idx, self.segment_start_idx:self.segment_end_idx]
            self.point_nodes[point_idx].baseline_optical_flow_world_traj = optical_flow_3d_trajs[point_idx]
        return optical_flow_3d_trajs


    
    def estimate_plane_rotation(self, 
                       depth_trunc=5.0,
                       z_up=True,
                       plane_estimation_kwargs={
                            "ransac_n": 3,
                            "num_iterations": 1000,
                            "distance_threshold": 0.01
                       }):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"
        assert(self.input_depth is not None), "Depth not loaded"
        
        o3d_pcd = O3DPointCloud()
        o3d_pcd.create_from_rgbd(
            self.input_image,
            self.input_depth,
            self.camera_intrinsics,
            depth_trunc=depth_trunc,
        )

        plane_estimation_result = o3d_pcd.plane_estimation(**plane_estimation_kwargs)
        
        T_xy_plane_align = estimate_rotation(plane_estimation_result["plane_model"], z_up=z_up)
        return T_xy_plane_align, plane_estimation_result

    def estimate_motion_traj_from_object(self, object_id, 
                                         use_visibility=True,
                                         skip_interval=1,
                                         select_subset=-1,
                                         mode="lie",
                                         regularization_weight_pos=1.0,
                                         regularization_weight_rot=1.0,
                                         num_max_iter=5,
                                         high_occlusion=False,
                                         optim_kwargs={
                                             "lr": 0.01, 
                                             "num_epochs": 1001,
                                             "verbose": True}):
        """Currently only assuming rigid body manipulation. """
        world_trajs = self.get_world_trajs(object_ids=[object_id])
        invalid_indices = self.get_invalid_world_trajs_indices_by_object(object_id=object_id)

        visibility = self.get_visibility_trajs(object_ids=[object_id])[:, -1]
        filter_visibility_indices = np.where(visibility < 0.5)[0].tolist()
        # print("Before visibility filtering: ", invalid_indices)
        if high_occlusion:
            invalid_indices = np.array(list(set(invalid_indices + filter_visibility_indices)))
        # print("Adding visibility filtering: ", filter_visibility_indices)

        # print("Invalid indices: ", invalid_indices)

        world_trajs = np.delete(world_trajs, invalid_indices, axis=0)
        all_keypoints = torch.tensor(world_trajs).float()
        # (NUM_OBJECTS, NUM_POINTS, 3) -> (NUM_POINTS, NUM_OBJECTS, 3)
        all_keypoints = all_keypoints.permute(1, 0, 2) 

        all_visibilities = None
        if use_visibility:
            all_visibilities = self.get_visibility_trajs(object_ids=[object_id])
            all_visibilities = np.delete(all_visibilities, invalid_indices, axis=0)
            all_visibilities = torch.tensor(all_visibilities).float()   
            all_visibilities = all_visibilities.permute(1, 0)
        # delete invalid indices of all_keypoints and all_visibilities

        all_keypoints = all_keypoints[::skip_interval]
        if use_visibility:
            all_visibilities = all_visibilities[::skip_interval]

        if select_subset > 0:
            all_keypoints = all_keypoints[:, :select_subset]
            if use_visibility:
                all_visibilities = all_visibilities[:, :select_subset]

        ORION_LOGGER.debug("Optimization array shape:")
        ORION_LOGGER.debug(f"Keypoint shape: {all_keypoints.shape}")
        if use_visibility:
            ORION_LOGGER.debug(f"Visibility shape: {all_visibilities.shape}")
        # if mode == "simple":
        #     bundle_optimization = optim_utils.SimpleBundleOptimization()
        #     optimized_transforms = bundle_optimization.optimize(all_keypoints, all_visibilities, optim_kwargs)
        # else:
            
        # with mp.Pool(8) as pool:
        #     bundle_optimization = optim_utils.LieBundleOptimization()
        #     optimized_transforms, optimized_translation, best_loss = bundle_optimization.optimize(all_keypoints, 
        #                                                     all_visibilities, 
        #                                                     # regularization_weight=regularization_weight, 
        #                                                     regularization_weight_pos=regularization_weight_pos,
        #                                                     regularization_weight_rot=regularization_weight_rot,
        #                                                     optim_kwargs=optim_kwargs)

        #     solutions = pool.map(solve, [Solver(points1_ext, points_2_uv, cmat) for _ in range(max_iter)])
        # best_solution = solutions[np.argmin([s.fun for s in solutions])]
        bundle_optimization = optim_utils.LieBundleOptimization()
        loss_threshold = 0.1
        final_best_loss = 1000
        final_optimized_transforms = None
        final_optimized_translation = None
        for _ in range(num_max_iter):
            optimized_transforms, optimized_translation, best_loss = bundle_optimization.optimize(all_keypoints, 
                                                            all_visibilities, 
                                                            regularization_weight_pos=regularization_weight_pos,
                                                            regularization_weight_rot=regularization_weight_rot,
                                                            optim_kwargs=optim_kwargs)
            if loss_threshold > best_loss:
                break
            else:
                if best_loss < final_best_loss:
                    final_best_loss = best_loss
                    final_optimized_transforms = optimized_transforms
                    final_optimized_translation = optimized_translation
                ORION_LOGGER.warning("Loss larger than threshold. Retrying ... ")
        if final_optimized_transforms is not None:
            optimized_transforms = final_optimized_transforms
            optimized_translation = final_optimized_translation


        return optimized_transforms, optimized_translation, best_loss, all_keypoints
                
    def draw_input_image(self, width=300, height=300):
        plotly_draw_image(self.input_image, width=width, height=height)

    def draw_overlay_image(self, mode, width=300, height=300, default_point_color=None):
        assert(mode in ["object", "point", "all"]), "mode must be either `object`, `point` or `all`"

        if mode == "object":
            ORION_LOGGER.debug("Plotting object overlay")
            overlay_image = overlay_xmem_mask_on_image(
                self.input_image, 
                self.input_annotation,
                use_white_bg=True,
                rgb_alpha=0.4)
            plotly_draw_seg_image(
                overlay_image,
                self.input_annotation, 
                width=width, 
                height=height)
        elif mode == "point":
            ORION_LOGGER.debug("Plotting point overlay")
            
            plotly_draw_image_with_object_keypoints(
                self.input_image, 
                self.get_point_list(mode="pixel"), 
                width=width, 
                height=height,
                default_point_color=default_point_color)
        elif mode == "all":
            ORION_LOGGER.debug("Plotting all overlay")
            overlay_image = overlay_xmem_mask_on_image(
                self.input_image, 
                self.input_annotation,
                use_white_bg=True,
                rgb_alpha=0.4)
            plotly_draw_image_with_object_keypoints(
                overlay_image, 
                self.get_point_list(mode="pixel"), 
                width=width, 
                height=height)
        else:
            raise NotImplementedError
        
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


    def set_camera_extrinsics(self, camera_extrinsics):
        self.camera_extrinsics = camera_extrinsics
    
    def set_camera_intrinsics(self, camera_intrinsics):
        self.camera_intrinsics = camera_intrinsics
    
    def load_depth(self, input_depth):
        self.input_depth = np.squeeze(np.ascontiguousarray(input_depth))
        # update the points
        self.create_world_points()

    def draw_rgb_depth(self, width=800, height=300):
        assert(self.input_depth is not None), "Depth not loaded"
        depth_in_rgb = depth_to_rgb(self.input_depth)
        side_by_side_img = np.concatenate((self.input_image, depth_in_rgb), axis=1)
        plotly_draw_image(side_by_side_img, width=width, height=height)

    def draw_scene_3d(self, 
                      draw_trajs=False,
                      draw_keypoints=False,
                      additional_points=None,
                      downsample=True,
                      depth_trunc=3.0,
                      no_background=False,
                      overlay_annotation=False,
                      additional_point_draw_lines=False,
                      use_white_bg=False,
                      skip_trajs=[],
                      skip_keypoints=[]
):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"

        rgb_img_input = self.input_image
        if overlay_annotation:
            rgb_img_input = overlay_xmem_mask_on_image(
                self.input_image, 
                self.input_annotation,
                use_white_bg=use_white_bg,
                rgb_alpha=0.4)
        pcd_points, pcd_colors = scene_pcd_fn(
            rgb_img_input=rgb_img_input,
            depth_img_input=self.input_depth,
            extrinsic_matrix=self.camera_extrinsics,
            intrinsic_matrix=self.camera_intrinsics,
            downsample=downsample,
            depth_trunc=depth_trunc,
        )

        marker_size = 3

        if draw_trajs:
            if additional_points is None:
                additional_points = self.get_world_trajs()
                additional_points = additional_points.reshape(self.num_objects, -1, 3)
                # skip some points using np delete
                additional_points = np.delete(additional_points, skip_trajs, axis=0)
                    
        elif draw_keypoints:
            if additional_points is None:
                additional_points = self.get_points_by_objects(object_ids=self.object_ids, mode="world")
                additional_points = np.array(additional_points)
                # skip some points using np delete
                additional_points = np.delete(additional_points, skip_keypoints, axis=0)
            marker_size = 8

        plotly_draw_3d_pcd(pcd_points, pcd_colors, addition_points=additional_points, marker_size=marker_size, no_background=no_background, additional_point_draw_lines=additional_point_draw_lines)
        return pcd_points, pcd_colors

    def get_objects_3d_points(self, object_id=None, filter=True, remove_outlier_kwargs={"nb_neighbors": 40, "std_ratio": 0.7}, downsample=True):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"

        if object_id is None:
            masked_depth = self.input_depth * (self.input_annotation > 0).astype(np.float32)
        else:
            masked_depth = self.input_depth * (self.input_annotation == object_id).astype(np.float32)
        pcd_points, pcd_colors = scene_pcd_fn(
            rgb_img_input=self.input_image,
            depth_img_input=masked_depth,
            extrinsic_matrix=self.camera_extrinsics,
            intrinsic_matrix=self.camera_intrinsics,
            downsample=downsample,
        )
        if filter:
            pcd_points, pcd_colors = remove_outlier(pcd_points, pcd_colors,
                                                    **remove_outlier_kwargs)
        return pcd_points, pcd_colors
    
    def draw_objects_3d(self, 
                        object_id=None, 
                        draw_trajs=False, 
                        draw_keypoints=False,
                        additional_points=None,
                        downsample=True,
                        no_background=False,
                        additional_point_draw_lines=False,
                        keypoints_rgb_str="(255, 0, 0)",
                        uniform_color=False):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"

        pcd_points, pcd_colors = self.get_objects_3d_points(object_id=object_id, downsample=downsample)
        if draw_trajs and draw_keypoints:
            ORION_LOGGER.warning("draw_trajs and draw_keypoints are both True, only draw_trajs will be used")

        marker_size = 3
        if draw_trajs:
            if additional_points is None:
                additional_points = self.get_world_trajs()
                additional_points = additional_points.reshape(self.num_objects, -1, 3)
        elif draw_keypoints:
            if additional_points is None:
                object_ids = self.object_ids if object_id is None else [object_id]
                additional_points = self.get_points_by_objects(object_ids=object_ids, mode="world")
                additional_points = np.squeeze(np.array(additional_points)).reshape(len(object_ids), -1, 3)
                ORION_LOGGER.debug(f"Additional points shape: {additional_points.shape}")
            marker_size = 10
        
        plotly_draw_3d_pcd(pcd_points, pcd_colors, addition_points=additional_points, marker_size=marker_size, no_background=no_background, additional_point_draw_lines=additional_point_draw_lines, default_rgb_str=keypoints_rgb_str, uniform_color=uniform_color)

        return pcd_points, pcd_colors
    

    def draw_objects_list_3d(self, 
                        object_ids=None, 
                        draw_trajs=False, 
                        draw_keypoints=False,
                        additional_points=None,
                        downsample=True,
                        no_background=False,
                        additional_point_draw_lines=False,
                        keypoints_rgb_str="(255, 0, 0)",
                        uniform_color=False):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"

        pcd_points_list = []
        pcd_colors_list = []
        for object_id in object_ids:
            pcd_points, pcd_colors = self.get_objects_3d_points(object_id=object_id, downsample=downsample)
            pcd_points_list.append(pcd_points)
            pcd_colors_list.append(pcd_colors)
        pcd_points = np.concatenate(pcd_points_list, axis=0)
        pcd_colors = np.concatenate(pcd_colors_list, axis=0)
        if draw_trajs and draw_keypoints:
            ORION_LOGGER.warning("draw_trajs and draw_keypoints are both True, only draw_trajs will be used")

        marker_size = 3
        if draw_trajs:
            if additional_points is None:
                additional_points = self.get_world_trajs()
                additional_points = additional_points.reshape(self.num_objects, -1, 3)
        elif draw_keypoints:
            if additional_points is None:
                object_ids = self.object_ids if object_id is None else [object_id]
                additional_points = self.get_points_by_objects(object_ids=object_ids, mode="world")
                additional_points = np.squeeze(np.array(additional_points)).reshape(len(object_ids), -1, 3)
                ORION_LOGGER.debug(f"Additional points shape: {additional_points.shape}")
            marker_size = 10
        
        plotly_draw_3d_pcd(pcd_points, pcd_colors, addition_points=additional_points, marker_size=marker_size, no_background=no_background, additional_point_draw_lines=additional_point_draw_lines, default_rgb_str=keypoints_rgb_str, uniform_color=uniform_color)

        return pcd_points, pcd_colors    


    def draw_points_3d(self, object_id=None):
        raise NotImplementedError

    def draw_dense_correspondence(self, other_graph, object_ids=[]):
        assert(self.task_id == other_graph.task_id), "Task ID must be the same, otherwise the correspondence is meaningless"
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        other_selected_point_nodes = other_graph.select_point_nodes(object_ids=object_ids)

        self_point_list = self.get_point_list(point_nodes=selected_point_nodes, mode="pixel")
        other_point_list = other_graph.get_point_list(point_nodes=other_selected_point_nodes, mode="pixel")

        plotly_draw_image_correspondences(self.input_image, self_point_list, other_graph.input_image, other_point_list)

    def estimate_depth(self):
        """This function is used when estimating depth from rgb images."""
        raise NotImplementedError

    def set_pixel_trajs(self, pixel_trajs):
        assert(pixel_trajs.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_pixel_traj = pixel_trajs[i]

    def create_world_points(self):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"
        points = create_point_clouds_from_keypoints(
            np.array(self.get_point_list(mode="pixel")),
            self.input_depth[..., np.newaxis],
            self.camera_intrinsics,
        )
        for i in range(len(self.point_nodes)):
            self.point_nodes[i].world_point = transform_points(self.camera_extrinsics, points[i:i+1])

    def set_world_trajs(self, world_trajs):
        assert(world_trajs.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_world_traj = world_trajs[i]
            point_node.world_point = world_trajs[i][0]

    def get_invalid_world_trajs_indices_by_object(self, object_id):
        point_nodes = self.select_point_nodes(object_ids=[object_id])
        invalid_indices = []
        for i, point_node in enumerate(point_nodes):
            if point_node.invalid_world_traj:
                invalid_indices.append(i)
        return invalid_indices


    def set_world_trajs_by_object(self, world_trajs, object_id, outlier_indices=[]):
        point_nodes = self.select_point_nodes(object_ids=[object_id])
        assert(world_trajs.shape[0] == len(point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(point_nodes):
            point_node.tracked_world_traj = world_trajs[i]
            point_nodes[i].invalid_world_traj = False

        for i in outlier_indices:
            point_nodes[i].invalid_world_traj = True

    def set_optical_flow_world_trajs_by_object(self, world_trajs, object_id, outlier_indices=[]):
        point_nodes = self.select_point_nodes(object_ids=[object_id])
        assert(world_trajs.shape[0] == len(point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(point_nodes):
            point_node.baseline_optical_flow_world_trajs = world_trajs[i]
            point_nodes[i].invalid_world_traj = False

        for i in outlier_indices:
            point_nodes[i].invalid_world_traj = True

    def set_visibility_trajs(self, visibility_traj):
        assert(visibility_traj.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_visibility_traj = visibility_traj[i]
            point_node.tracked_visibility = visibility_traj[i][0]

    def set_visibility_trajs_by_object(self, object_id, visibility_traj):
        point_nodes = self.select_point_nodes(object_ids=[object_id])
        assert(visibility_traj.shape[0] == len(point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(point_nodes):
            point_node.tracked_visibility_traj = visibility_traj[i]
            point_node.tracked_visibility = visibility_traj[i][0]

    def set_visibility(self, visibility):
        assert(visibility.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_visibility = visibility[i]          

    def get_pixel_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        pixel_trajs = []
        for point_node in selected_point_nodes:
            pixel_trajs.append(point_node.tracked_pixel_traj)
        return np.stack(pixel_trajs, axis=0)
    
    def get_world_trajs(self, object_ids=[]):
        """_summary_

        Args:
            object_ids (list, optional): _description_. Defaults to [].

        Returns:
           np.ndarray: (NUM_OBJECTS, NUM_POINTS, 3)
        """
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        world_trajs = []
        for point_node in selected_point_nodes:
            world_trajs.append(point_node.tracked_world_traj)
        return np.stack(world_trajs, axis=0)
    
    def get_optical_flow_world_trajs(self, object_ids=[]):
        """_summary_

        Args:
            object_ids (list, optional): _description_. Defaults to [].

        Returns:
           np.ndarray: (NUM_OBJECTS, NUM_POINTS, 3)
        """
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        world_trajs = []
        for point_node in selected_point_nodes:
            world_trajs.append(point_node.baseline_optical_flow_world_traj)
        return np.stack(world_trajs, axis=0)    
    
    def get_visibility_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        visibility = []
        for point_node in selected_point_nodes:
            visibility.append(point_node.tracked_visibility_traj)
        return np.stack(visibility, axis=0)
    
    def get_visibility(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        visibility = []
        for point_node in selected_point_nodes:
            visibility.append(point_node.tracked_visibility)
        return np.stack(visibility, axis=0)

    @property
    def num_objects(self):
        return len(self.object_nodes)

    @property
    def num_points(self):
        return len(self.point_nodes)
    
    @property
    def object_ids(self):
        return [object_node.object_id for object_node in self.object_nodes]
    
    def get_pixel_point_by_node(self, point_node_idx):
        return self.point_nodes[point_node_idx].pixel_point
    
    def get_pixel_point_by_object_point_node(self, object_id, idx):
        object_node = self.object_nodes[object_id - 1]
        point_node_idx = object_node.get_point(idx)
        return self.point_nodes[point_node_idx].pixel_point
    
    def get_world_point_by_node(self, point_node_idx):
        return self.point_nodes[point_node_idx].world_point

    def get_world_point_by_object_node(self, object_id, idx):
        object_node = self.object_nodes[object_id - 1]
        point_node_idx = object_node.get_point(idx)
        return self.point_nodes[point_node_idx].world_point
    
    def correspondence_heatmap(self, aff, point, other_graph):
        patch_x, patch_y = math.floor(point[0] / self.input_image.shape[1] * aff.shape[0]), math.floor(point[1] / self.input_image.shape[0] * aff.shape[1])

        selected_aff = aff[patch_y, patch_x]

        argmax_y, argmax_x = np.unravel_index(selected_aff.argmax(), selected_aff.shape)

        normalized_heatmap = 255 - cv2.normalize(selected_aff, None, alpha=0, beta=255, 
                                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        colormap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
        colormap = resize_image_to_same_shape(colormap, other_graph.input_image)

        heatmap_image_2 = cv2.addWeighted(other_graph.input_image, 0.7, colormap, 0.3, 0)
        heatmap_image_2 = cv2.circle(heatmap_image_2, (argmax_x, argmax_y), 15, (0, 0, 255), -1)
        reference_image = cv2.circle(np.copy(self.input_image), (point[0], point[1]), 15, (0, 0, 255), -1)
        return reference_image, heatmap_image_2
    
    # def get_current_hand_traj(self, target_transform, subgoal_transform, interaction_points):
    #     thumb_tip_trajs, thumb_dip_trajs, index_tip_trajs, index_dip_trajs = self.get_thumb_index_trajectories()

    #     hand_traj = np.stack([thumb_tip_trajs, index_tip_trajs], axis=1)
    #     print(hand_traj.shape)
    #     new_trajs = []
    #     for k in range(2):
    #         hand_init_point = hand_traj[0][k]
    #         hand_goal_point = hand_traj[-1][k]

    #         traj_processor = SimpleTrajProcessor(hand_init_point, hand_goal_point)
    #         normalized_traj = traj_processor.normalize_traj(hand_traj[:, k])
    #         new_init_point = interaction_points[k]
    #         offset = (object_2_pcd_points.mean(axis=0) - demo_object_2_pcd_points.mean(axis=0))
    #         new_goal_point = transform_point_clouds(subgoal_transform, (hand_goal_point + offset).reshape(1, 3))

    #         new_goal_point = new_goal_point.reshape(3)
    #         new_traj = traj_processor.unnormalize_traj(normalized_traj, new_init_point, new_goal_point)
    #         new_trajs.append(new_traj)

    #     new_traj = np.stack(new_trajs, axis=0)
    #     return new_traj
    

def run_optimization(bundle_optimization, all_keypoints, all_visibilities, regularization_weight_pos, regularization_weight_rot, optim_kwargs):
    # Run the optimization task
    return bundle_optimization.optimize(all_keypoints, all_visibilities, regularization_weight_pos=regularization_weight_pos, regularization_weight_rot=regularization_weight_rot, optim_kwargs=optim_kwargs)

def parallel_optimization(all_keypoints, all_visibilities, regularization_weight_pos, regularization_weight_rot, optim_kwargs, num_runs):
    # Create a LieBundleOptimization instance
    bundle_optimization = optim_utils.LieBundleOptimization()
    mp.set_start_method('spawn', force=True)

    # Setup a pool of processes
    with torch.multiprocessing.Pool(processes=num_runs) as pool:
        # Prepare arguments for each process
        args = [(bundle_optimization, all_keypoints, all_visibilities, regularization_weight_pos, regularization_weight_rot, optim_kwargs) for _ in range(num_runs)]

        # Run the optimization in parallel
        results = pool.starmap(run_optimization, args)

    return results

