# mp
import multiprocessing as mp
from orion.utils.misc_utils import *
from orion.utils.tap_utils import *
from orion.algos.hoig import HandObjectInteractionGraph
from orion.algos.tap_segmentation import TAPSegmentation
from orion.algos.temporal_segments import TemporalSegments
from orion.utils.correspondence_utils import CorrespondenceModel
from orion.utils.traj_utils import SimpleTrajProcessor

from orion.utils.o3d_utils import *
from orion.utils.log_utils import get_orion_logger
from orion.algos.retargeter_wrapper import Retargeter
from orion.algos.grasp_primitives import GraspPrimitive
from orion.utils.robosuite_utils import *

class HumanVideoHOIG:
    def __init__(self):
        self.temporal_segments = TemporalSegments()
        self.waypoints_info = []
        self.hoigs = []
        self.human_video_annotation_path = ""

        self.smplh_traj = np.array([])
        self.retargeted_ik_traj = np.array([])
        
        self.object_names = []

    def generate_from_human_video(self, human_video_annotation_path, zero_pose_name="ready", video_smplh_ratio=1.0, use_smplh=True):
        
        self.human_video_annotation_path = human_video_annotation_path

        tap_segmentation = TAPSegmentation()
        tap_segmentation.load(human_video_annotation_path)
        self.temporal_segments = tap_segmentation.temporal_segments

        with open(os.path.join(human_video_annotation_path, "waypoints_info.json"), "r") as f:
            self.waypoints_info = json.load(f)    
        self.object_names = get_object_names_from_annotation(human_video_annotation_path)

        retargeter = None
        if use_smplh:
            self.smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)
            len_smplh = len(self.smplh_traj)
            len_video = self.temporal_segments.segments[-1].end_idx
            video_smplh_ratio = len_video / len_smplh
            print("video_smplh_ratio", video_smplh_ratio)
            retargeter = Retargeter(example_data=self.smplh_traj[0])

        grasp_dict_l = GraspPrimitive()
        grasp_dict_r = GraspPrimitive()

        self.hoigs = []
        for i in range(len(self.temporal_segments.segments)):
            hoig = HandObjectInteractionGraph()
            hoig.create_from_human_video(human_video_annotation_path, 
                                         self.temporal_segments.segments[i].start_idx, 
                                         self.temporal_segments.segments[i].end_idx-1,
                                         segment_idx=i,
                                         retargeter=retargeter,
                                         grasp_dict_l=grasp_dict_l,
                                         grasp_dict_r=grasp_dict_r,
                                         calibrate_grasp=(i == 0),
                                         zero_pose_name=zero_pose_name,
                                         video_smplh_ratio=video_smplh_ratio,
                                         use_smplh=use_smplh)
            self.hoigs.append(hoig)

        # get reference object and manipulate object names


        print("generated HOIGs from human video, total_num=", len(self.hoigs), self.temporal_segments)
    
    def get_graph(self, idx):
        return self.hoigs[idx]
    
    @property
    def num_graphs(self):
        return len(self.hoigs)

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
            ORION_LOGGER.debug(f"Current graph: {graph_id}")

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

                ORION_LOGGER.debug(f"object_id: {object_id} | v:  {v_mean} | v_std: {v_std}")

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

            ORION_LOGGER.debug(f"contact_states: {graph_in_query.contact_states}")
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

    def get_retargeted_ik_traj(self, 
                               offset={"link_RArm7": [0, 0, 0]}, 
                               interpolation_steps=-1,
                               interpolation_type='linear'):
        
        retargeter = Retargeter(example_data=self.smplh_traj[0])

        ik_traj = []
        for idx, hoig in enumerate(self.hoigs):
            hoig.get_retargeted_ik_traj(retargeter=retargeter,
                                        offset=offset, 
                                        num_waypoints=1 if self.waypoints_info[idx] == 'Target Only' else hoig.segment_length//2,
                                        interpolation_steps=interpolation_steps, 
                                        interpolation_type=interpolation_type)
            ik_traj.append(hoig.retargeted_ik_traj)
        whole_traj = np.concatenate(ik_traj, axis=0)
        self.retargeted_ik_traj = whole_traj
        return whole_traj, ik_traj
    
    def get_retargeted_ik_traj_with_grasp_primitive(self, 
                                                    offset={"link_RArm7": [0, 0, 0]},
                                                    zero_pose_name="ready", 
                                                    interpolation_steps=-1, 
                                                    interpolation_type='linear'):
        
        retargeter = Retargeter(example_data=self.smplh_traj[0])
        grasp_dict_l = GraspPrimitive()
        grasp_dict_r = GraspPrimitive()
        
        ik_traj = []
        for idx, hoig in enumerate(self.hoigs):
            hoig.get_retargeted_ik_traj_with_grasp_primitive(retargeter=retargeter,
                                                             grasp_dict_l=grasp_dict_l,
                                                             grasp_dict_r=grasp_dict_r,
                                                             calibrate_grasp=(idx == 0),
                                                             zero_pose_name=zero_pose_name,
                                                             offset=offset, 
                                                             num_waypoints=1 if self.waypoints_info[idx] == 'Target Only' else hoig.segment_length//2,
                                                             interpolation_steps=interpolation_steps, 
                                                             interpolation_type=interpolation_type)
            ik_traj.append(hoig.retargeted_ik_traj)

        # fill in middle of grasp primitive change
        new_traj = []
        grasp_change_steps = 60
        for i in range(len(ik_traj)):
            grasp_tar = ik_traj[i][0]
            grasp_lst = ik_traj[i - 1][-1] if i > 0 else ik_traj[i][0]
            grasp_traj = []
            for j in range(grasp_change_steps):
                grasp_traj.append(grasp_lst + (grasp_tar - grasp_lst) * (j / grasp_change_steps))
            new_traj.append(np.array(grasp_traj))
            new_traj.append(ik_traj[i])
        whole_traj = np.concatenate(new_traj, axis=0)

        self.retargeted_ik_traj = whole_traj
        return whole_traj, new_traj
    
    def get_action_with_grasp_primitive(self, args, step, obs, traj, **kwargs):
        target_joint_pos = np.zeros(32)

        init_interpolate_steps = 30
        if step < init_interpolate_steps:
            # interpolate to traj[0]
            ed = calculate_target_qpos(traj[0])
            st = np.array([
                -0.0035970834452681436, 0.011031227286351492, -0.01311470003464996, 0.0, 0.0, 0.0, 0.8511509067155127, 1.310805039853726, -0.7118440391862395, -0.536551596928798, 0.02341464067352966, -0.23317144423063796, -0.0803808564555934, 0.18086797377837605, -1.5034221574091646, -0.15101789788918812, 0.00014316406250000944, -0.07930486850248092, -0.1222325540688668, -0.2801763429367678,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ])
            target_joint_pos = st + (ed - st) * (step / init_interpolate_steps)
        else:
            if step == init_interpolate_steps:
                print("init interpolation finished. replay begin!")
            # head, torso, and arms
            target_joint_pos = calculate_target_qpos(traj[min(step - init_interpolate_steps, len(traj) - 1)])

        if step % 100 == 0:
            print(f"step: {step} finished.")

        action = calculate_action_from_target_joint_pos(obs, target_joint_pos)
        return action

    def replay_traj_in_robosuite(self, max_steps=500, traj=None):
        args = argparse.Namespace()
        args.environment = "HumanoidSimple"
        args.save_path = os.path.join(self.human_video_annotation_path, "sim_output.mp4")

        offset = [-0., -0.0, 0.0]
        whole_traj = self.retargeted_ik_traj if traj is None else traj
        launch_simulator(args, 
                        get_action_func=self.get_action_with_grasp_primitive, 
                        max_steps=max_steps, 
                        offset=offset, 
                        traj=whole_traj)
    
    def get_num_segments(self):
        return len(self.hoigs)
    
    def get_all_segments_info(self):
        segments = []
        for idx, segment in enumerate(self.temporal_segments.segments):
            segments.append(convert_to_json_serializable({
                'start_idx': segment.start_idx,
                'end_idx': segment.end_idx,
                'waypoint_info': self.waypoints_info[idx],
                'grasp_type': self.hoigs[idx].grasp_type,
                'hand_type': self.hoigs[idx].hand_type,
                'moving_arm': self.hoigs[idx].moving_arm,
                'smplh_traj': convert_to_json_serializable(self.hoigs[idx].smplh_traj),
                'objects': self.object_names
            }))
            # print(segments[-1])
        return segments
    
    def visualize_plan(self, no_smplh=False):
        video_seq = get_video_seq_from_annotation(self.human_video_annotation_path)

        initial_images = []
        for segment in self.temporal_segments.segments:
            initial_images.append(video_seq[segment.start_idx])
        video_writer = VideoWriter(self.human_video_annotation_path, video_name=f"plan_vis.mp4", fps=30)
        for i in range(len(video_seq)):
            image_frame = []
            for idx, segment in enumerate(self.temporal_segments.segments):
                if segment.in_duration(i):
                    img = video_seq[i].copy()
                elif i >= segment.end_idx:
                    img = video_seq[segment.end_idx].copy()
                else:
                    img = video_seq[segment.start_idx].copy()

                self.add_text_on_image(img, f"Segment {segment.start_idx}-{segment.end_idx}", pos=(10,30), color=(0,0,0), fontsize=0.8)
                self.add_text_on_image(img, self.waypoints_info[idx], color=(0,50,0), pos=(10,60), fontsize=0.7)
                if not no_smplh:
                    self.add_text_on_image(img, f"left: {self.hoigs[idx].grasp_type[0]} right: {self.hoigs[idx].grasp_type[1]}", color=(0,50,100), pos=(10,90), fontsize=0.8)
                image_frame.append(img)
            
            image_frame = np.concatenate(image_frame, axis=1)
            video_writer.append_image(image_frame)
        video_writer.save(bgr=False)
    
    def add_text_on_image(self, img, text, pos=(10,30), color=(255,255,255), fontsize=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = fontsize
        fontColor = color
        lineType = 2
        cv2.putText(img, text, 
            pos, 
            font, 
            fontScale,
            fontColor,
            lineType)
        return img
