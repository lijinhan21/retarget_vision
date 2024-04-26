import argparse
import numpy as np
import os
import ruptures as rpt
import matplotlib.pyplot as plt

from easydict import EasyDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import init_path
from orion.utils.misc_utils import *
from orion.algos.tap_segmentation import TAPSegmentation
from orion.algos.hand_object_detector import HandObjectDetector

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    args = parser.parse_args()

    tap_segmentation = TAPSegmentation()
    tap_segmentation.load(args.annotation_folder)

    # extract keyframe images
    video_seq = get_video_seq_from_annotation(args.annotation_folder)
    keyframe_indices = [0]
    print("num_segments:", len(tap_segmentation.temporal_segments.segments))
    for seg in tap_segmentation.temporal_segments.segments:
        keyframe_indices.append(seg.end_idx)
    os.makedirs(os.path.join(args.annotation_folder, 'keyframes'), exist_ok=True)
    print("keyframe_indices:", keyframe_indices, len(keyframe_indices))
    for i in range(len(keyframe_indices)):
        keyframe = video_seq[keyframe_indices[i]]
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.annotation_folder, f'keyframes/{i}.png'), keyframe)

    # analyze hand-object contacts for each key frame, and visualize in images
    hand_object_detector = HandObjectDetector()
    for i in range(len(keyframe_indices)):
        obj_dets, hand_dets = hand_object_detector.detect(os.path.join(args.annotation_folder, f'keyframes/{i}.png'), 
                                                          vis=True, 
                                                          save_path=os.path.join(args.annotation_folder, f'keyframes/hand_object_dets/'))
        hand_info = hand_object_detector.parse_hand_info(hand_dets)
        print(f"Hand info for keyframe {i}: {hand_info}")

if __name__ == "__main__":
    main()