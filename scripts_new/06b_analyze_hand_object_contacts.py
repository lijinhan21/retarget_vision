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

    hand_object_detector = HandObjectDetector()

    # extract keyframe images
    video_seq = get_video_seq_from_annotation(args.annotation_folder)
    print("num_segments:", len(tap_segmentation.temporal_segments.segments))
    os.makedirs(os.path.join(args.annotation_folder, 'keyframes'), exist_ok=True)
    
    all_segments_info = []
    for idx, seg in enumerate(tap_segmentation.temporal_segments.segments):
        os.makedirs(os.path.join(args.annotation_folder, f'keyframes/{idx}'), exist_ok=True)
        
        # sample several keyframes from each segment
        keyframe_indices = np.linspace(seg.start_idx, seg.end_idx, 10).astype(int)
        for i in range(len(keyframe_indices)):
            keyframe = video_seq[keyframe_indices[i]]
            keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.annotation_folder, f'keyframes/{idx}/{i}.png'), keyframe)
        
        # analyze hand-object contacts for each key frame, and visualize in images
        all_res = []
        for i in range(len(keyframe_indices)):
            obj_dets, hand_dets = hand_object_detector.detect(os.path.join(args.annotation_folder, f'keyframes/{idx}/{i}.png'), 
                                                              vis=True, 
                                                              save_path=os.path.join(args.annotation_folder, f'keyframes/{idx}/hand_object_dets/'))
            hand_info = hand_object_detector.parse_hand_info(hand_dets)
            # print(f"Hand info for keyframe {i}: {hand_info}")
            all_res.append(hand_info)

        # combine all res
        segment_hand_info = {
            'left': {'in_contact': False, 'contact_type': 'none'},
            'right': {'in_contact': False, 'contact_type': 'none'}
        }
        num_left = {}
        num_right = {}
        for hand_info in all_res:
            left_key = f'''{hand_info['left']['in_contact']}_{hand_info['left']['contact_type']}'''
            right_key = f'''{hand_info['right']['in_contact']}_{hand_info['right']['contact_type']}'''
            if left_key not in num_left:
                num_left[left_key] = 0
            if right_key not in num_right:
                num_right[right_key] = 0
            num_left[left_key] += 1
            num_right[right_key] += 1
        # foung bigest number of left and right hand
        max_left = 0
        max_right = 0
        for key, value in num_left.items():
            if value > max_left:
                max_left = value
                segment_hand_info['left']['in_contact'], segment_hand_info['left']['contact_type'] = key.split('_')
        for key, value in num_right.items():
            if value > max_right:
                max_right = value
                segment_hand_info['right']['in_contact'], segment_hand_info['right']['contact_type'] = key.split('_')

        print(f"Segment {idx} hand info: {segment_hand_info}")
        print("Left hand use grasp pose?", segment_hand_info['left']['contact_type'] == 'portable')
        print("Right hand use grasp pose?", segment_hand_info['right']['contact_type'] == 'portable')

        all_segments_info.append(segment_hand_info)
    
    with open(os.path.join(args.annotation_folder, 'hand_object_contacts.json'), 'w') as f:
        json.dump(all_segments_info, f)

if __name__ == "__main__":
    main()