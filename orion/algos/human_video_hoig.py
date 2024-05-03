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

class HumanVideoHOIG:
    def __init__(self):
        self.temporal_segments = TemporalSegments()
        self.waypoints_info = []
        self.hoigs = []
        self.human_video_annotation_path = ""

        self.smplh_traj = np.array([])
        self.retargeted_ik_traj = np.array([])

    def generate_from_human_video(self, human_video_annotation_path, video_smplh_ratio=1.0, use_smplh=True):
        
        self.human_video_annotation_path = human_video_annotation_path

        tap_segmentation = TAPSegmentation()
        tap_segmentation.load(human_video_annotation_path)
        self.temporal_segments = tap_segmentation.temporal_segments

        with open(os.path.join(human_video_annotation_path, "waypoints_info.json"), "r") as f:
            self.waypoints_info = json.load(f)

        if use_smplh:
            self.smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)

        self.hoigs = []
        for i in range(len(self.temporal_segments.segments)):
            hoig = HandObjectInteractionGraph()
            hoig.create_from_human_video(human_video_annotation_path, 
                                         self.temporal_segments.segments[i].start_idx, 
                                         self.temporal_segments.segments[i].end_idx-1,
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
                                        num_waypoints=3 if self.waypoints_info[idx] == 'Target Only' else hoig.segment_length//2,
                                        interpolation_steps=interpolation_steps, 
                                        interpolation_type=interpolation_type)
            ik_traj.append(hoig.retargeted_ik_traj)
        whole_traj = np.concatenate(ik_traj, axis=0)
        self.retargeted_ik_traj = whole_traj
        return whole_traj, ik_traj
    
    def get_retargeted_ik_traj_with_grasp_primitive(self, 
                                                    offset={"link_RArm7": [0, 0, 0]}, 
                                                    interpolation_steps=-1, 
                                                    interpolation_type='linear'):
        
        retargeter = Retargeter(example_data=self.smplh_traj[0])

        ik_traj = []
        for idx, hoig in enumerate(self.hoigs):
            hoig.get_retargeted_ik_traj_with_grasp_primitive(retargeter=retargeter,
                                                             offset=offset, 
                                                             num_waypoints=3 if self.waypoints_info[idx] == 'Target Only' else hoig.segment_length//2,
                                                             interpolation_steps=interpolation_steps, 
                                                             interpolation_type=interpolation_type)
            ik_traj.append(hoig.retargeted_ik_traj)

        # fill in middle of grasp primitive change
        new_traj = []
        for i in range(len(ik_traj)):
            grasp_tar = ik_traj[i][0]
            grasp_lst = ik_traj[i - 1][-1] if i > 0 else ik_traj[i][0]
            grasp_traj = []
            for j in range(70):
                grasp_traj.append(grasp_lst + (grasp_tar - grasp_lst) * (j / 70))
            new_traj.append(np.array(grasp_traj))
            new_traj.append(ik_traj[i])
        whole_traj = np.concatenate(new_traj, axis=0)

        self.retargeted_ik_traj = whole_traj
        return whole_traj, new_traj
    
    def get_num_segments(self):
        return len(self.hoigs)
    
    def visualize_plan(self):
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

                self.add_text_on_image(img, f"Segment {segment.start_idx}-{segment.end_idx}", color=(0,0,0))
                self.add_text_on_image(img, self.waypoints_info[idx], color=(0,0,255), pos=(50,70))
                image_frame.append(img)
            
            image_frame = np.concatenate(image_frame, axis=1)
            video_writer.append_image(image_frame)
        video_writer.save(bgr=False)
    
    def add_text_on_image(self, img, text, pos=(10,30), color=(255,255,255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = color
        lineType = 2
        cv2.putText(img, text, 
            pos, 
            font, 
            fontScale,
            fontColor,
            lineType)
        return img
