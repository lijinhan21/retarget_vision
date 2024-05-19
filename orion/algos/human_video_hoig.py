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
        print("generated HOIGs from human video, total_num=", len(self.hoigs), self.temporal_segments)
    
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
