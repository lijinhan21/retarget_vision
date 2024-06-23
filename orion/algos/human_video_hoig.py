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
from orion.utils.gpt4v_utils import encode_imgs_from_path, json_parser
from orion.algos.gpt4v import GPT4V

class HumanVideoHOIG:
    def __init__(self):
        self.temporal_segments = TemporalSegments()
        self.waypoints_info = []
        self.hoigs = []
        self.human_video_annotation_path = ""

        self.smplh_traj = np.array([])
        self.retargeted_ik_traj = np.array([])
        
        self.object_names = []
        self.object_id_to_name = {}

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
        estimated_extrinsics = None
        for i in range(len(self.temporal_segments.segments)):
            hoig = HandObjectInteractionGraph()
            hoig.create_from_human_video(human_video_annotation_path, 
                                         self.temporal_segments.segments[i].start_idx, 
                                         self.temporal_segments.segments[i].end_idx-1,
                                         segment_idx=i,
                                         extrinsics=estimated_extrinsics,
                                         retargeter=retargeter,
                                         grasp_dict_l=grasp_dict_l,
                                         grasp_dict_r=grasp_dict_r,
                                         calibrate_grasp=(i == 0),
                                         zero_pose_name=zero_pose_name,
                                         video_smplh_ratio=video_smplh_ratio,
                                         use_smplh=use_smplh)
            self.hoigs.append(hoig)
            estimated_extrinsics = hoig.camera_extrinsics

        # get reference object and manipulate object names
        self.get_object_id_to_name()
        self.plan_inference()

        print("generated HOIGs from human video, total_num=", len(self.hoigs), self.temporal_segments)
    
    def get_graph(self, idx):
        return self.hoigs[idx]
    
    @property
    def num_graphs(self):
        return len(self.hoigs)
    
    def get_object_id_to_name(self):
        first_graph = self.get_graph(0)
        for object_id in first_graph.object_ids:
            rgb_img = first_graph.get_objects_2d_image(object_id)
            cv2.imwrite(os.path.join(self.human_video_annotation_path, f"object_{object_id}.png"), rgb_img)
            
            vlm = GPT4V()
            vlm.begin_new_dialog()
            text_description = f'''Please Identify the object in the image below. You need to choose from the following objects: {self.object_names}.''' + '''
Your output format:
```json
{
    "object_name": "OBJECT_NAME",
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
You should output the name of the object in a string "OBJECT_NAME" that can be easily parsed by Python. The name is a string, e.g., "apple", "pen", "keyboard", etc.
'''
            text_prompt_list = [text_description]
            base64_img_list = [vlm.preprocess_image(rgb_img, is_rgb=True, encoding='jpg')]
            text_response = vlm.run(text_prompt_list, base64_img_list)
            json_data = json_parser(text_response)

            if json_data is not None and 'object_name' in json_data:
                self.object_id_to_name[object_id] = json_data["object_name"]
            
        print("object_id_to_name", self.object_id_to_name)

    def vlm_get_reference_object(self, graph_id):
        current_graph = self.get_graph(graph_id)

        if current_graph.get_manipulate_object_id() < 0:
            print("no manipulation object in this frame. skip reference object selection.")
            return

        manipulate_object_name = self.object_id_to_name[current_graph.get_manipulate_object_id()]

        vlm = GPT4V()
        vlm.begin_new_dialog()
        text_description = f'''The following images shows a manipulation motion, where the human is manipulating the object {manipulate_object_name}.
Please identify the reference object in the image below, which could be an object on which to place {manipulate_object_name}, or an object with {manipulate_object_name} is interacting with.
Note that there may not necessarily have an reference object, as sometimes human may just interacting with the object itself, like throwing it, or spinning it around.
You need to first identify is there is a reference object. If so, you need to output the reference object's name chosen from the following objects: {self.object_names}.''' + '''
Your output format is:

```json
{
    "reference_object_name": "REFERENCE_OBJECT_NAME" or None,
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
'''
        text_prompt_list = [text_description]
        base64_img_list = [[vlm.preprocess_image(img, is_rgb=True, encoding='jpg') for img in current_graph.representative_images]]

        text_response = vlm.run(text_prompt_list, base64_img_list)
        json_data = json_parser(text_response)

        if json_data['reference_object_name'] == 'None':
            current_graph.set_reference_object_id(-1)
        else:
            for object_id, object_name in self.object_id_to_name.items():
                if object_name == json_data['reference_object_name']:
                    current_graph.set_reference_object_id(object_id)
                    break
        print("reference object id", current_graph.get_reference_object_id(), 'name', json_data['reference_object_name'])

    def vlm_get_manipulate_object(self, graph_id):
        current_graph = self.get_graph(graph_id)
        
        vlm = GPT4V()
        vlm.begin_new_dialog()

        text_description = f'''The following images shows a manipulation motion, where the human is manipulating an object. Your task is to determine which object is being manipulated in the images below. You need to choose from the following objects: {self.object_names}.
Tips: the manipulated object is the object that the human is interacting with, such as picking up, moving, or pressing, and it is in contact with human's {current_graph.moving_arm} hand.''' + '''
Your output format is:

```json
{
    "manipulate_object_name": "MANIPULATE_OBJECT_NAME",
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
'''
        text_prompt_list = [text_description]
        base64_img_list = [[vlm.preprocess_image(img, is_rgb=True, encoding='jpg') for img in current_graph.representative_images]]

        text_response = vlm.run(text_prompt_list, base64_img_list)
        json_data = json_parser(text_response)

        for object_id, object_name in self.object_id_to_name.items():
            if object_name == json_data['manipulate_object_name']:
                current_graph.set_manipulate_object_id(object_id)
                break
        print("manipulate object id", current_graph.get_manipulate_object_id(), 'name', json_data['manipulate_object_name'])


    def plan_inference(self, velocity_threshold=0.8, target_dist_threshold=0.05, target_intersection_threshold=600):
        """This is a function to infer the important information in a plan. Specifically, we will get the following information from this function: 
            1. Manipulate object sequence, which specifies the object that the robot should manipulate in each step
            2. Reference object sequence, which specifies the object that the robot should use as reference in each step

            -1 in the object sequence means that no object is selected for manipulation or reference.
        Args:
            velocity_threshold (_type_, optional): _description_. Defaults to 1..

        Returns:
            _type_: _description_
        """
        for graph_id in range(self.num_graphs):
            graph_in_query = self.get_graph(graph_id)
            assert(type(graph_in_query) == HandObjectInteractionGraph)

            # the object with the highest velocity is selected as the object to manipulate
            candidate_objects_to_move = []
            v_mean_list = []
            v_std_list = []
            for object_id in graph_in_query.object_ids:
                point_nodes = graph_in_query.select_point_nodes(object_ids=[object_id])
                points = graph_in_query.get_point_list(point_nodes=point_nodes)
                world_trajs = graph_in_query.get_pixel_trajs(object_ids=[object_id])
                all_visibilities = graph_in_query.get_visibility_trajs(object_ids=[object_id])
                traj_diff = world_trajs[:, 1:, :] - world_trajs[:, :-1, :]
                confidence = all_visibilities[:, 1:] * all_visibilities[:, :-1]

                traj_diff = traj_diff * confidence[:, :, None]

                v_mean = np.mean(np.linalg.norm(traj_diff, axis=-1))
                v_std = np.std(np.linalg.norm(traj_diff, axis=-1))

                if v_mean > velocity_threshold:
                    v_mean_list.append(v_mean)
                    v_std_list.append(v_std)
                    candidate_objects_to_move.append(object_id)

                print("velocity of object", object_id, ":", v_mean, v_std)

            lr = 0 if graph_in_query.moving_arm == "L" else 1
            if len(candidate_objects_to_move) == 0:
                if (graph_in_query.hand_type[lr] == 'open') or (graph_id == 0):
                    graph_in_query.set_manipulate_object_id(-1)
                else:
                    print("No moving objects found. Will try to determine the object being manipulated using hand-object contacts.")
                    self.vlm_get_manipulate_object(graph_id)
            else:
                if (len(candidate_objects_to_move) == 1) or (np.std(v_mean_list) < 0.1):
                    graph_in_query.set_manipulate_object_id(candidate_objects_to_move[0])
                else:
                    # sort candidate objects to move by v_mean
                    candidate_objects_to_move = [x for _, x in sorted(zip(v_mean_list, candidate_objects_to_move))]
                    graph_in_query.set_manipulate_object_id(candidate_objects_to_move[-1])

            # Get point clouds for all objects
            pcd_list = []
            for object_id in graph_in_query.object_ids:
                pcd_array, _ = graph_in_query.get_objects_3d_points(object_id=object_id, filter=False, downsample=False)
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
                    print(i+1, ", ", j+1, " | ", len(intersections))
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

            print("manipulate object id", graph_in_query.get_manipulate_object_id())
            print("contact states", contact_states)

            rgb_img = graph_in_query.get_objects_2d_image(graph_in_query.get_manipulate_object_id())
            os.makedirs(os.path.join(self.human_video_annotation_path, "objects"), exist_ok=True)
            cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"manipulate_object_{graph_id}.png"), rgb_img)

        # decide reference objects between a pair of graphs
        for graph_id in range(self.num_graphs - 1):
            current_graph = self.get_graph(graph_id)
            next_graph = self.get_graph(graph_id + 1)

            contact_states = current_graph.contact_states
            current_graph.set_reference_object_id(-1)

            if current_graph.get_manipulate_object_id() > 0:
                contact_states = next_graph.contact_states
                manipulate_object_id = current_graph.get_manipulate_object_id()
                
                print("graph id=", graph_id, "manipulate object id=", manipulate_object_id, "next frame contact states=", contact_states)

                if len(contact_states) > 0:
                    for contact_state in contact_states:
                        if manipulate_object_id in contact_state:
                            temp_contact_state = list(contact_state)
                            temp_contact_state.remove(manipulate_object_id)
                            current_graph.set_reference_object_id(temp_contact_state[0])
                            break
            else:
                # This is free motion. Then reference object is the next frame's manipulate object
                current_graph.set_reference_object_id(next_graph.get_manipulate_object_id())

            if current_graph.get_reference_object_id() < 0:
                print("cannot determine reference object based on contacts. trying to call gpt4v")
                if current_graph.get_manipulate_object_id() > 0:
                    self.vlm_get_reference_object(graph_id)

            print("reference object id", current_graph.get_reference_object_id())

            rgb_img = current_graph.get_objects_2d_image(current_graph.get_reference_object_id())
            cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"reference_object_{graph_id}.png"), rgb_img)

            if current_graph.get_manipulate_object_id() > 0 and current_graph.get_reference_object_id() > 0:
                print("calculating desired distance")

                pcd_manipulate, _ = next_graph.get_objects_3d_points(object_id=current_graph.get_manipulate_object_id(), filter=False, downsample=False)
                pcd_reference, _ = next_graph.get_objects_3d_points(object_id=current_graph.get_reference_object_id(), filter=False, downsample=False)

                manipulate_center = np.mean(np.array(pcd_manipulate), axis=0)
                reference_center = np.mean(np.array(pcd_reference), axis=0)
                translation = manipulate_center - reference_center

                current_graph.target_translation_btw_objects = translation
                print("translation", translation)

        # decide reference objects for the last graph
        last_graph = self.get_graph(self.num_graphs - 1)
        last_graph.set_reference_object_id(-1)
        # print("reference object id", last_graph.get_reference_object_id())

        rgb_img = last_graph.get_objects_2d_image(last_graph.get_reference_object_id())
        cv2.imwrite(os.path.join(self.human_video_annotation_path, "objects", f"reference_object_{self.num_graphs-1}.png"), rgb_img)

        return
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
                'manipulate_object': self.object_id_to_name[self.hoigs[idx].manipulate_object_id] if self.hoigs[idx].manipulate_object_id > 0 else "None",
                'reference_object': self.object_id_to_name[self.hoigs[idx].reference_object_id] if self.hoigs[idx].reference_object_id > 0 else "None",
                'target_translation': self.hoigs[idx].target_translation_btw_objects.tolist(),
                # 'smplh_traj': convert_to_json_serializable(self.hoigs[idx].smplh_traj),
                # 'objects': self.object_names
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
                # self.add_text_on_image(img, self.waypoints_info[idx], color=(0,50,0), pos=(10,60), fontsize=0.7)
                if not no_smplh:
                    self.add_text_on_image(img, f"left: {self.hoigs[idx].grasp_type[0]} right: {self.hoigs[idx].grasp_type[1]}", color=(0,50,100), pos=(10,60), fontsize=0.8)
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
