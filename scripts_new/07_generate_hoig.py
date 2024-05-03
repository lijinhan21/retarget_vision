import argparse
import numpy as np
import os
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
    human_video_hoig.generate_from_human_video(args.annotation_folder, 1.2, use_smplh=False if args.no_smplh else True)

    if not args.no_smplh:
        print("shape of retargeted traj", human_video_hoig.get_retargeted_ik_traj()[0].shape)
        human_video_hoig.get_retargeted_ik_traj_with_grasp_primitive()
    
    human_video_hoig.visualize_plan(no_smplh=args.no_smplh)

if __name__ == "__main__":
    main()