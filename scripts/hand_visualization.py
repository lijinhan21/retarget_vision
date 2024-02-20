import cv2
import argparse
import numpy as np
from plotly import graph_objects as go

from PIL import Image
from pathlib import Path

import init_path
from orion.utils.misc_utils import get_hamer_result, VideoWriter, plotly_draw_image_with_hand_keypoints

def draw_points_opencv(image, points):
    # Assuming points is a list of tuples (x, y)

    wrist_idx = 0
    thumb_indices = [1, 2, 3, 4]
    index_indices = [5, 6, 7, 8]
    middle_indices = [9, 10, 11, 12]
    ring_indices = [13, 14, 15, 16]
    pinky_indices = [17, 18, 19, 20]
    indices_dict = {
        "thumb": thumb_indices,
        "index": index_indices,
        "middle": middle_indices,
        "ring": ring_indices,
        "pinky": pinky_indices
    }
    colors_dict = {
        "wrist": (255, 0, 0), # "red",
        "thumb": (0, 255, 0), # "blue",
        "index": (0, 0, 255),# "green",
        "middle": (122, 122, 0),# "yellow",
        "ring": (0, 122, 122),# "orange",
        "pinky": (122, 0, 122), # "purple"
    }    

    for finger_name in indices_dict.keys():
        indices = indices_dict[finger_name]
        for idx in range(len(indices) - 1):
            point1 = points[indices[idx]].astype(np.int32)
            point2 = points[indices[idx + 1]].astype(np.int32)
            cv2.line(image, point1, point2, colors_dict[finger_name], thickness=2)
        for idx in range(len(indices)):
            point = points[indices[idx]].astype(np.int32)
            cv2.circle(image, point, radius=5, color=colors_dict[finger_name], thickness=-1)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", type=str, required=True)
    args = parser.parse_args()

    hamer_result, hamer_result_path = get_hamer_result(args.annotation_path)

    # Load output images

    recon_video_writer = VideoWriter(args.annotation_path, video_name="hamer_recon.mp4", fps=30, save_video=True)
    print(hamer_result_path)
    for img_path in sorted(Path(hamer_result_path).glob("*_all.jpg")):
        img = cv2.imread(str(img_path))
        recon_video_writer.append_image(img)
    recon_video_writer.save()
    hand_pose_video_writer = VideoWriter(args.annotation_path, video_name="hamer_hand_pose.mp4", fps=30, save_video=True)
    print(hamer_result_path)
    for idx, img_path in enumerate(sorted(Path(hamer_result_path).glob("*_all.jpg"))):
        img = cv2.imread(str(img_path))
        hand_pose_keypoints = hamer_result["vitpose_detections"][idx][:, 0:2]
        new_img = draw_points_opencv(img, hand_pose_keypoints)
        hand_pose_video_writer.append_image(new_img)
        # img_buf = plotly_draw_image_with_hand_keypoints(img,, save_to_buffer=True)
        # hand_pose_video_writer.append_image(np.array(Image.open(img_buf)))

    hand_pose_video_writer.save()

    thumb_index_joints = hamer_result["hand_joints_seq"][0].squeeze()
    fig = go.Figure()

    distances = np.linalg.norm(thumb_index_joints[:, 4, :] - thumb_index_joints[:, 8, :], 2, axis=-1)
    fig.add_trace(go.Scatter(x=np.arange(len(distances)), y=distances))

    fig.update_layout(
        title="Distance between the thumb and the index finger",
        xaxis_title="Frame",
        yaxis_title="Distance",
    )
    fig.show()

if __name__ == "__main__":
    main()