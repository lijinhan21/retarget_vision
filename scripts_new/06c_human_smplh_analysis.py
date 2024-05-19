import argparse
import numpy as np
import os
import ruptures as rpt
import matplotlib.pyplot as plt

import init_path
from orion.utils.misc_utils import *
from orion.algos.hoig import HandObjectInteractionGraph
from orion.algos.human_video_hoig import HumanVideoHOIG

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument('--smplh-path', type=str, help='Path to the smplh trajectory file.')
    args = parser.parse_args()

    # TODO: call Georgios' code here
    # TODO: normalize the smplh_traj

    # TODO: save the normalized smplh_traj to os.path.join(args.annotation_folder, "smplh_traj.pkl") 

    # temporary: copy args.smplh_path in side args.annotation_folder
    smplh_path = os.path.join('datasets/smplh', args.smplh_path)
    os.system(f'''cp {smplh_path} {os.path.join(args.annotation_folder, "smplh_traj.pkl")}''')

if __name__ == "__main__":
    main()