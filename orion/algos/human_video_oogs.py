
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from hausdorff import hausdorff_distance

import robosuite.utils.transform_utils as robosuite_transform

# mp
import multiprocessing as mp
from orion.utils.misc_utils import *
from orion.utils.tap_utils import *
from orion.algos.oog import OpenWorldObjectSceneGraph, OOGMode
from orion.algos.tap_segmentation import TAPSegmentation, OpticalFlowSegmentation
from orion.algos.temporal_segments import TemporalSegments
from orion.utils.correspondence_utils import CorrespondenceModel
from orion.utils.traj_utils import SimpleTrajProcessor

from orion.utils.o3d_utils import *

def identify_outliers_with_silhouette(original_traj, n_clusters=2):
    traj = original_traj[:, 1:, :] - original_traj[:, :-1, :]
    N, T, _ = traj.shape
    reshaped_traj = traj.reshape(N, -1)  # Reshape to 2D for simplicity

    # Standardize the features
    scaler = StandardScaler()
    scaled_traj = scaler.fit_transform(reshaped_traj)

    # Cluster the trajectories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_traj)

    # Calculate silhouette scores
    silhouette_vals = silhouette_samples(scaled_traj, cluster_labels)

    # Identify trajectories with low silhouette scores as outliers
    outlier_indices = np.where(silhouette_vals < np.mean(silhouette_vals) - np.std(silhouette_vals))[0]
    print(f"Detected outlier indices: {outlier_indices}")
    return outlier_indices

class HumanVideoOOGs():
    def __init__(self, 
                 task_name=""):
        self.task_name = task_name
        self.correspondence_model = None
        self.init()

    def init(self, debug_mode=True):
        self.human_video_annotation_path = None
        self.mode = "lab"

        self.temporal_segments = TemporalSegments()
        self.human_object_graphs = []

        self.debug_mode = debug_mode

        self.remove_outlier_config = {
            "nb_neighbors": 30, 
            "std_ratio": 0.7
        }
        self.global_registration_config = {
            "voxel_size": 0.01,
            # "max_correspondence_distance": 0.01, 
            # "max_iter": 1000
            }

    def reset(self):
        self.init()
        
    def save(self, save_path):
        save_graphs = []
        for graph in self.human_object_graphs:
            save_graphs.append(graph.to_dict())
        save_dict = {
            "task_name": self.task_name,
            "human_video_annotation_path": self.human_video_annotation_path,
            "mode": self.mode,
            "temporal_segments": self.temporal_segments.to_torch_tensor(),
            "human_object_graphs": save_graphs,
        }
        torch.save(save_dict, save_path)

    def load(self, load_path):
        load_dict = torch.load(load_path)
        self.task_name = load_dict["task_name"]
        self.human_video_annotation_path = load_dict["human_video_annotation_path"]
        self.mode = load_dict["mode"]
        self.temporal_segments = TemporalSegments()
        self.temporal_segments.load(load_dict["temporal_segments"])
        self.human_object_graphs = []
        for graph_dict in load_dict["human_object_graphs"]:
            graph = OpenWorldObjectSceneGraph(debug_mode=self.debug_mode)
            graph.from_dict(graph_dict)
            self.human_object_graphs.append(graph)

    def set_correspondence_model(self, correspondence_model: CorrespondenceModel):
        self.correspondence_model = correspondence_model

    def generate_from_human_video(self, 
                                 human_video_annotation_path,
                                 mode="lab",
                                 filter_annotation=False,
                                 plane_estimation_depth_trunc=5.0,
                                 plane_estimation_kwargs={
                                        "ransac_n": 3,
                                        "num_iterations": 1000,
                                        "distance_threshold": 0.01
                                 },
                                 baseline_optical_flow=False
                                 ):
        self.mode = mode
        self.human_video_annotation_path = human_video_annotation_path

        tap_results = get_tracked_points_annotation(human_video_annotation_path)

        tap_segmentation = TAPSegmentation()
        tap_segmentation.load(human_video_annotation_path)
        self.temporal_segments = tap_segmentation.temporal_segments

        if baseline_optical_flow:
            tap_segmentation = OpticalFlowSegmentation()
            tap_segmentation.load(human_video_annotation_path)
            self.temporal_segments = tap_segmentation.temporal_segments

        if self.debug_mode:
            print("temporal_segments: ", self.temporal_segments)

        self.human_object_graphs = []
        for segment in self.temporal_segments:
            human_object_graph = OpenWorldObjectSceneGraph(debug_mode=self.debug_mode)
            human_object_graph.generate_from_human_demo(
                human_video_annotation_path,
                mode=mode,
                segment_start_idx=segment.start_idx,
                segment_end_idx=segment.end_idx-1,
                previous_graph=None if len(self.human_object_graphs) == 0 else self.human_object_graphs[-1],
                filter_annotation=filter_annotation,
                plane_estimation_depth_trunc=plane_estimation_depth_trunc,
                plane_estimation_kwargs=plane_estimation_kwargs,
            )
            self.human_object_graphs.append(human_object_graph)

        human_object_graph = OpenWorldObjectSceneGraph(debug_mode=self.debug_mode)
        human_object_graph.generate_from_human_demo(
            human_video_annotation_path,
            mode=mode,
            segment_start_idx=segment.end_idx-1,
            segment_end_idx=segment.end_idx,
            previous_graph=None if len(self.human_object_graphs) == 0 else self.human_object_graphs[-1],
            filter_annotation=filter_annotation,
            baseline_optical_flow=baseline_optical_flow
        )
        self.human_object_graphs.append(human_object_graph)
        print("Generated # of human_object_graphs: ", len(self.human_object_graphs))


    def skip_free_motion(self, current_idx):
        oog_mode_seq = self.get_oog_mode_sequence()
        while oog_mode_seq[current_idx] == OOGMode.FREE_MOTION and current_idx < self.num_graphs - 1:
            # print(current_idx, ", ", current_idx + 1, " : ", self.check_graph_contact_equal(current_idx, current_idx + 1))
            if self.check_graph_contact_equal(current_idx, current_idx + 1):
                current_idx += 1
            else:
                break
        return current_idx

    def find_matching_oog_idx(self, robot_object_graph: OpenWorldObjectSceneGraph):
        """Find the matching human object graph for the robot object graph."""

        query_idx = 0
        
        matched = True
        while matched and query_idx < self.num_graphs - 1:
            graph_in_query = self.get_graph(query_idx)

            matching_condition = True
            for contact_state in graph_in_query.contact_states:
                # print(contact_state, " : ", robot_object_graph.contact_states)
                matching_condition = matching_condition & self.check_contact(contact_state, robot_object_graph.contact_states)
            print(query_idx, ": ", matching_condition)
            if matching_condition:
                new_query_idx = self.skip_free_motion(query_idx)
                if new_query_idx == query_idx:
                    query_idx += 1
                else:
                    query_idx = new_query_idx
            else:
                matched = False
                print("Quitting: ", query_idx)
            print(query_idx)
            if query_idx >= self.num_graphs - 1:
                break
        if query_idx != self.num_graphs - 1:
            query_idx = max(0, query_idx - 1)
        return query_idx
    
    def check_contact(self, query_state, state_list):
        ordered_query_state = list(query_state)
        ordered_query_state.sort()
        ordered_query_state = tuple(ordered_query_state)
        for state in state_list:
            ordered_state = list(state)
            ordered_state.sort()
            ordered_state = tuple(ordered_state)
            if ordered_state == ordered_query_state:
                return True
        return False
    
    def check_contact_list(self, state_list_1, state_list_2):
        equal = True
        equal = equal & (len(state_list_1) == len(state_list_2))
        for state in state_list_1:
            equal = equal & self.check_contact(state, state_list_2)
        return equal
    
    def check_graph_contact_equal(self, graph_idx_1, graph_idx_2):
        contact_states_1 = self.get_graph(graph_idx_1).contact_states
        contact_states_2 = self.get_graph(graph_idx_2).contact_states
        return self.check_contact_list(contact_states_1, contact_states_2)
    
    @property
    def num_graphs(self):
        return len(self.human_object_graphs)
    
    def get_graph(self, idx):
        if idx < 0 or idx >= len(self.human_object_graphs):
            print("Invalid idx our of range: ", idx)
            return None
        return self.human_object_graphs[idx]
    
    def compute_subgoal(self, 
                        matched_idx, 
                        robot_object_graph,
                        manipulate_object_id, 
                        reference_object_id,
                        deviation_angle_threshold=30,
                        high_occlusion=False,
                        remove_outlier_config=None,
                        skip_trivial_solution=False,
                        global_registration_config=None):
        
        if remove_outlier_config is None:
            remove_outlier_config = self.remove_outlier_config
        if (matched_idx == self.num_graphs - 1):
            print("You've already completed the task!")
            # The last human object graph
            return None
        matched_human_object_graph = self.get_graph(matched_idx)
        next_human_object_graph = self.get_graph(matched_idx + 1)

        subgoal_points = next_human_object_graph.get_points_by_objects(object_ids=[manipulate_object_id], mode="world")
        subgoal_visibility = next_human_object_graph.get_visibility(object_ids=[manipulate_object_id])

        reference_points = np.array(next_human_object_graph.get_points_by_objects(object_ids=[reference_object_id], mode="world"))
        reference_visibility = next_human_object_graph.get_visibility(object_ids=[reference_object_id])

        new_reference_points = np.squeeze(np.array(robot_object_graph.get_points_by_objects(object_ids=[reference_object_id], mode="world")))
        new_reference_visibility = robot_object_graph.get_visibility(object_ids=[reference_object_id])

        reference_pcd, _ = next_human_object_graph.get_objects_3d_points(object_id=reference_object_id, 
                                                                filter=True, remove_outlier_kwargs=self.remove_outlier_config)
        new_reference_pcd, _ = robot_object_graph.get_objects_3d_points(object_id=reference_object_id,
                                                                filter=True, remove_outlier_kwargs=self.remove_outlier_config)
        
        target_pcd, _ = matched_human_object_graph.get_objects_3d_points(object_id=manipulate_object_id, 
                                                                filter=True, remove_outlier_kwargs=self.remove_outlier_config)
        new_target_pcd, _ = robot_object_graph.get_objects_3d_points(object_id=manipulate_object_id,
                                                                filter=True, remove_outlier_kwargs=self.remove_outlier_config)
        
        new_reference_offset = (new_reference_pcd.mean(axis=0) - reference_pcd.mean(axis=0))
        new_target_offset = (new_target_pcd.mean(axis=0) - target_pcd.mean(axis=0))

        reference_pcd = reference_pcd + new_reference_offset
        target_pcd = target_pcd + new_target_offset
        # new_reference_pcd = new_reference_pcd + (reference_pcd.mean(axis=0) - new_reference_pcd.mean(axis=0))
        # new_target_pcd = new_target_pcd + (target_pcd.mean(axis=0) - new_target_pcd.mean(axis=0))
        if global_registration_config is None:
            global_registration_config = self.global_registration_config

        best_subgoal_hausdoff = np.inf
        best_target_hausdoff = np.inf
        best_subgoal_transform = None
        best_target_transform = None
        num_max_iter = 10
        sampled_new_reference_pcd = new_reference_pcd
        sampled_new_target_pcd = new_target_pcd

        # # Example usage
        # best_subgoal_transform, best_target_transform = parallel_global_registration(num_max_iter, reference_pcd, new_reference_pcd, target_pcd, new_target_pcd, global_registration_config)

        def deviate_from_z_axis(transformation_matrix):
            rotation_axis = robosuite_transform.quat2axisangle(robosuite_transform.mat2quat(transformation_matrix[:3, :3]))
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            dot_product = np.dot(rotation_axis, np.array([0, 0, 1]))
            # Calculate the angle (in radians) using the arccosine of the dot product
            angle_radians = np.arccos(dot_product)

            # Convert the angle to degrees
            angle_degrees = np.degrees(angle_radians)
            return angle_degrees
       
        count = 0

        while best_subgoal_transform is None or best_target_transform is None:

            # can be commented out
            if best_subgoal_transform is None and best_subgoal_transform is None and not skip_trivial_solution:
                # it means that this is the first time. Let's initialize the values with trivial solutions, where only translation is involved and no rotation is needed.
                subgoal_transform = np.eye(4)
                # subgoal_transform[:3, 3] = -new_reference_offset
                target_transform = np.eye(4)
                # target_transform[:3, 3] = -new_target_offset
                # estimated_reference_pcd = transform_point_clouds(subgoal_transform, reference_pcd)
                # estimated_target_pcd = transform_point_clouds(target_transform, target_pcd)
                subgoal_hausdorff_dist = hausdorff_distance(reference_pcd, sampled_new_reference_pcd)
                target_hausdorff_dist = hausdorff_distance(target_pcd, sampled_new_target_pcd)

                if subgoal_hausdorff_dist < best_subgoal_hausdoff:
                    best_subgoal_hausdoff = subgoal_hausdorff_dist
                    best_subgoal_transform = subgoal_transform
                if target_hausdorff_dist < best_target_hausdoff:
                    best_target_hausdoff = target_hausdorff_dist
                    best_target_transform = target_transform


            for i in range(num_max_iter):
                subgoal_transform = global_registration(reference_pcd, new_reference_pcd, voxel_size=global_registration_config["voxel_size"] * np.random.uniform())
                estimated_reference_pcd = transform_point_clouds(subgoal_transform, reference_pcd)
                if len(estimated_reference_pcd) > len(new_reference_pcd):
                    estimated_reference_pcd = random_subsampling(estimated_reference_pcd, len(new_reference_pcd))
                else:
                    sampled_new_reference_pcd = random_subsampling(new_reference_pcd, len(reference_pcd))
                subgoal_hausdorff_dist = hausdorff_distance(estimated_reference_pcd, sampled_new_reference_pcd)
                
                if subgoal_hausdorff_dist < best_subgoal_hausdoff and deviate_from_z_axis(subgoal_transform) < deviation_angle_threshold:
                    print("target_hausdorff_dist: ", subgoal_hausdorff_dist)
                    best_subgoal_hausdoff = subgoal_hausdorff_dist
                    best_subgoal_transform = subgoal_transform

                target_transform = global_registration(target_pcd, new_target_pcd, voxel_size=global_registration_config["voxel_size"] * np.random.uniform())
                estimated_target_pcd = transform_point_clouds(target_transform, target_pcd)
                if len(estimated_target_pcd) > len(new_target_pcd):
                    estimated_target_pcd = random_subsampling(estimated_target_pcd, len(new_target_pcd))
                else:
                    sampled_new_target_pcd = random_subsampling(new_target_pcd, len(target_pcd))
                target_hausdorff_dist = hausdorff_distance(estimated_target_pcd, sampled_new_target_pcd)
                if target_hausdorff_dist < best_target_hausdoff and deviate_from_z_axis(target_transform) < deviation_angle_threshold:
                    print("target_hausdorff_dist: ", target_hausdorff_dist)
                    best_target_hausdoff = target_hausdorff_dist
                    best_target_transform = target_transform
            count += 1

        subgoal_transform = best_subgoal_transform
        target_transform = best_target_transform

        # subgoal_transform = global_registration(reference_pcd, new_reference_pcd, voxel_size=global_registration_config["voxel_size"])

        # target_transform = global_registration(target_pcd, new_target_pcd, voxel_size=global_registration_config["voxel_size"])

        new_subgoal_points = []
        subgoal_points = subgoal_points + new_reference_offset
        for point in subgoal_points:
            new_subgoal_points.append(transform_points(subgoal_transform, point.reshape(1, 3)))
        new_subgoal_points = np.concatenate(new_subgoal_points, axis=0).reshape(-1, 3)


        reference_trajs = matched_human_object_graph.get_world_trajs(object_ids=[manipulate_object_id])

        new_trajs = []
        for i in range(len(subgoal_points)):
            ref_traj = reference_trajs[i]
            traj_processor = SimpleTrajProcessor(ref_traj[0], ref_traj[-1])
            normalized_traj = traj_processor.normalize_traj(ref_traj)
            new_init_point_candidate = matched_human_object_graph.get_world_point_by_object_node(object_id=manipulate_object_id, idx=i) + new_target_offset
            new_init_point = transform_points(target_transform, new_init_point_candidate.reshape(1, 3))
            new_goal_point = new_subgoal_points[i]
            new_traj = traj_processor.unnormalize_traj(normalized_traj, new_init_point.reshape(-1), new_goal_point.reshape(-1))
            new_trajs.append(new_traj)

        new_trajs = np.stack(new_trajs, axis=0)
        outlier_indices = identify_outliers_with_silhouette(new_trajs)
        selected_trajs = np.delete(new_trajs, outlier_indices, axis=0)

        # TODO: Handle the case of trajectory already being filtered --- they shouldn't go into estimation of transformation as well.
        robot_object_graph.set_world_trajs_by_object(new_trajs, object_id=manipulate_object_id, outlier_indices=outlier_indices)

        new_subgoal_visibility_trajs = np.ones_like(matched_human_object_graph.get_visibility_trajs()).astype(np.bool)
        robot_object_graph.set_visibility_trajs(new_subgoal_visibility_trajs)
        
        if high_occlusion:
            filter_visibility = next_human_object_graph.get_visibility(object_ids=[manipulate_object_id])
            traj_len = new_subgoal_visibility_trajs.shape[1]
            filter_visibility_trajs = np.repeat(filter_visibility[:, None], traj_len, axis=1)
            # print(filter_visibility_trajs.shape)
            robot_object_graph.set_visibility_trajs_by_object(object_id=manipulate_object_id, visibility_traj=filter_visibility_trajs)

            
        return subgoal_transform, target_transform


    def debug_visualization(self):
        raise NotImplementedError
    
    def get_oog_mode_sequence(self):
        """_summary_

        Returns:
            list of OOGMode: A list of OOGMode that specifies the interaction mode. You need to convert your self to get an integer sequence of the trajectory. 
        """
        mode_seq = []
        for graph in self.human_object_graphs:
            mode_seq.append(graph.get_oog_mode())
        return mode_seq
    
    def get_manipulate_object_seq(self):
        seq = []
        for graph in self.human_object_graphs:
            seq.append(graph.get_manipulate_object_id())
        return seq
    
    def get_reference_object_seq(self):
        seq = []
        for graph in self.human_object_graphs:
            seq.append(graph.get_reference_object_id())
        return seq
    
    def plan_inference(self, velocity_threshold=1., target_dist_threshold=0.01, target_intersection_threshold=1000):
        """This is a function to infer the important information in a plan. Specifically, we will get the following information from this function: 
            1. OOG mode sequence, which specifies the interaction mode (freespace, move with gripper closed, move with gripper open)
            2. Manipulate object sequence, which specifies the object that the robot should manipulate in each step
            3. Reference object sequence, which specifies the object that the robot should use as reference in each step

            -1 in the object sequence means that no object is selected for manipulation or reference.
        Args:
            velocity_threshold (_type_, optional): _description_. Defaults to 1..

        Returns:
            _type_: _description_
        """
        for graph_id in range(self.num_graphs):
            graph_in_query = self.get_graph(graph_id)
            print("Current graph: ", graph_id)
            num_objects = graph_in_query.num_objects

            # graph_in_query.draw_scene_3d(draw_trajs=True)
            candidate_objects_to_move = []
            v_mean_list = []
            v_std_list = []
            for object_id in graph_in_query.object_ids:
                point_nodes = graph_in_query.select_point_nodes(object_ids=[object_id])
                points = graph_in_query.get_point_list(point_nodes=point_nodes)
                # print("object_id: ", object_id, " | points: ", graph_in_query.object_nodes[object_id-1].points)
                # world_trajs =  graph_in_query.get_world_trajs(object_ids=[object_id])
                world_trajs = graph_in_query.get_pixel_trajs(object_ids=[object_id])
                all_visibilities = graph_in_query.get_visibility_trajs(object_ids=[object_id])
                # print(world_trajs.shape, all_visibilities.shape)
                traj_diff = world_trajs[:, 1:, :] - world_trajs[:, :-1, :]
                # traj_diff = world_trajs[:, 1:, :] - world_trajs[:, :1, :]
                confidence = all_visibilities[:, 1:] * all_visibilities[:, :-1]

                traj_diff = traj_diff * confidence[:, :, None]
                # print(traj_diff.shape, confidence.shape)
                # v_mean = np.mean(np.sum((np.linalg.norm(traj_diff, axis=-1)), axis=1) / traj_diff.shape[1], axis=0) 
                # v_std = np.std(np.sum((np.linalg.norm(traj_diff, axis=-1)), axis=1) / traj_diff.shape[1], axis=0)
                v_mean = np.mean(np.linalg.norm(traj_diff, axis=-1))
                v_std = np.std(np.linalg.norm(traj_diff, axis=-1))
                print("object_id: ", object_id, " | v: ", v_mean, " | v_std: ", v_std)

                if v_mean > velocity_threshold:
                    v_mean_list.append(v_mean)
                    v_std_list.append(v_std)
                    candidate_objects_to_move.append(object_id)
            if len(candidate_objects_to_move) == 0:
                graph_in_query.set_oog_mode(OOGMode.FREE_MOTION)
            else:
                if (len(candidate_objects_to_move) == 1) or (np.std(v_mean_list) < 0.1):
                    graph_in_query.set_manipulate_object_id(candidate_objects_to_move[0])
                else:
                    # sort candidate objects to move by v_mean
                    candidate_objects_to_move = [x for _, x in sorted(zip(v_mean_list, candidate_objects_to_move))]
                    graph_in_query.set_manipulate_object_id(candidate_objects_to_move[-1])

            pcd_list = []
            for object_id in graph_in_query.object_ids:
                pcd_array, _ = graph_in_query.get_objects_3d_points(object_id=object_id,
                                                                            filter=False)
                pcd = create_o3d_from_points_and_color(pcd_array)
                pcd_list.append(pcd)

            dists = []
            contact_states = []

            dist_threshold = target_dist_threshold
            intersection_threshold = target_intersection_threshold
            max_intersection = 0

            dists = []
            contact_states = []

            objects_in_contact = []
            for i in range(len(pcd_list)):
                for j in range(i+1, len(pcd_list)):
                    dists = pcd_list[i].compute_point_cloud_distance(pcd_list[j])
                    intersections = [d for d in dists if d < dist_threshold]
                    # print(i+1, ", ", j+1, " | ", len(intersections))
                    if len(intersections) > intersection_threshold:
                        contact_states.append((i+1, j+1))
                        objects_in_contact.append(i+1)
                        objects_in_contact.append(j+1)

            objects_in_contact = list(set(objects_in_contact))
            if graph_id > 0:
                prev_graph = self.get_graph(graph_id - 1)
                if len(prev_graph.contact_states) > 0:
                    for contact_state in prev_graph.contact_states:
                        if contact_state[0] not in objects_in_contact and contact_state[1] not in objects_in_contact:
                            contact_states.append(contact_state)

            
            graph_in_query.contact_states = contact_states

            print("contact_states: ", graph_in_query.contact_states)
            if graph_in_query.get_manipulate_object_id() > 0:
                manipulate_object_id = graph_in_query.get_manipulate_object_id()
                if len(contact_states) > 0:

                    for contact_state in contact_states:
                        if manipulate_object_id in contact_state:
                            temp_contact_state = list(contact_state)
                            temp_contact_state.remove(manipulate_object_id)
                            graph_in_query.set_reference_object_id(temp_contact_state[0])
                            break
                graph_in_query.decide_gripper_action()
            # TODO: decide the reference object
        # decide reference objects between a pair of graphs
        for graph_id in range(self.num_graphs - 1):
            current_graph = self.get_graph(graph_id)
            next_graph = self.get_graph(graph_id + 1)
            contact_states = graph_in_query.contact_states
            if current_graph.get_manipulate_object_id() > 0:
                contact_states = next_graph.contact_states
                manipulate_object_id = current_graph.get_manipulate_object_id()
                if len(contact_states) > 0:
                    for contact_state in contact_states:
                        if manipulate_object_id in contact_state:
                            temp_contact_state = list(contact_state)
                            temp_contact_state.remove(manipulate_object_id)
                            current_graph.set_reference_object_id(temp_contact_state[0])
                            break
        if not self.check_graph_contact_equal(self.num_graphs-2, self.num_graphs-1):
            current_graph = self.get_graph(self.num_graphs-2)
            next_graph = self.get_graph(self.num_graphs-1)        
            current_graph.contact_states = next_graph.contact_states
        return {
            "oog_mode_seq": self.get_oog_mode_sequence(),
            "manipulate_object_seq": self.get_manipulate_object_seq(),
            "reference_object_seq": self.get_reference_object_seq(),
        }


def transform_point_clouds(transformation, points):
        new_points = transformation @ np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
        new_points = new_points[:3, :].T
        return new_points

def random_subsampling(point_cloud, num_points):
    # point_cloud is an Nx3 numpy array, num_points is the desired number of points
    indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
    subsampled = point_cloud[indices, :]
    return subsampled

# def worker(reference_pcd, new_reference_pcd, target_pcd, new_target_pcd, global_registration_config):
#     # Logic from inside your loop
#     print("start worker")
#     print("ln0")
#     subgoal_transform = global_registration(reference_pcd, new_reference_pcd, voxel_size=global_registration_config["voxel_size"])
#     estimated_reference_pcd = transform_point_clouds(subgoal_transform, reference_pcd)
#     sampled_new_reference_pcd = new_reference_pcd
#     if len(estimated_reference_pcd) > len(new_reference_pcd):
#         estimated_reference_pcd = random_subsampling(estimated_reference_pcd, len(new_reference_pcd))
#     else:
#         sampled_new_reference_pcd = random_subsampling(new_reference_pcd, len(reference_pcd))
#     subgoal_hausdorff_dist = hausdorff_distance(estimated_reference_pcd, sampled_new_reference_pcd)

#     target_transform = global_registration(target_pcd, new_target_pcd, voxel_size=global_registration_config["voxel_size"] * np.random.uniform())



#     estimated_target_pcd = transform_point_clouds(target_transform, target_pcd)
#     sampled_new_target_pcd = new_target_pcd
#     if len(estimated_target_pcd) > len(new_target_pcd):
#         estimated_target_pcd = random_subsampling(estimated_target_pcd, len(new_target_pcd))
#     else:
#         sampled_new_target_pcd = random_subsampling(new_target_pcd, len(target_pcd))

#     target_hausdorff_dist = hausdorff_distance(estimated_target_pcd, sampled_new_target_pcd)

#     print("end worker")
#     return subgoal_hausdorff_dist, subgoal_transform, target_hausdorff_dist, target_transform

# def parallel_global_registration(num_max_iter, reference_pcd, new_reference_pcd, target_pcd, new_target_pcd, global_registration_config):
#     best_subgoal_hausdoff = float('inf')
#     best_target_hausdoff = float('inf')
#     best_subgoal_transform = None
#     best_target_transform = None

#     # results = worker(reference_pcd, new_reference_pcd, target_pcd, new_target_pcd, global_registration_config)

#     with mp.Pool() as pool:
#         results = pool.starmap(worker, [(reference_pcd, new_reference_pcd, target_pcd, new_target_pcd, global_registration_config) for _ in range(num_max_iter)])

#     for subgoal_hausdorff_dist, subgoal_transform, target_hausdorff_dist, target_transform in results:
#         if subgoal_hausdorff_dist < best_subgoal_hausdoff:
#             best_subgoal_hausdoff = subgoal_hausdorff_dist
#             best_subgoal_transform = subgoal_transform
#         if target_hausdorff_dist < best_target_hausdoff:
#             print("target_hausdorff_dist: ", target_hausdorff_dist)
#             best_target_hausdoff = target_hausdorff_dist
#             best_target_transform = target_transform

#     return best_subgoal_transform, best_target_transform