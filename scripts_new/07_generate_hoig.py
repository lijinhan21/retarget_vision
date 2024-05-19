import argparse
import numpy as np
import os
import json
import ruptures as rpt

import init_path
from orion.utils.misc_utils import *
from orion.algos.hoig import HandObjectInteractionGraph
from orion.algos.human_video_hoig import HumanVideoHOIG

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument('--no-smplh', action='store_true', default=False, help='Do not use SMPLH.')
    args = parser.parse_args()

    # hoig = HandObjectInteractionGraph()
    # hoig.create_from_human_video(args.annotation_folder, 0, 100)

    human_video_hoig = HumanVideoHOIG()
    human_video_hoig.generate_from_human_video(args.annotation_folder, 
                                               zero_pose_name="ready",
                                               video_smplh_ratio=1.0, 
                                               use_smplh=False if args.no_smplh else True)

    # if not args.no_smplh:
    #     # whole_traj, ik_traj = human_video_hoig.get_retargeted_ik_traj()
    #     whole_traj, ik_traj = human_video_hoig.get_retargeted_ik_traj_with_grasp_primitive()
    #     # print("shape of retargeted traj with grasp primitive", human_video_hoig.retargeted_ik_traj.shape)
    #     print("shape of each step:")
    #     for i in ik_traj:
    #         print(i.shape)

    #     human_video_hoig.replay_traj_in_robosuite(max_steps=500, traj=whole_traj)
    
    human_video_hoig.visualize_plan(no_smplh=args.no_smplh)

    segments_info = human_video_hoig.get_all_segments_info()
    with open(os.path.join(args.annotation_folder, "segments_info.json"), "w") as f:
        json.dump(segments_info, f)

if __name__ == "__main__":
    main()