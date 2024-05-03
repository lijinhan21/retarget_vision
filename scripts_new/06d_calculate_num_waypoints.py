import argparse
import numpy as np
import os
import ruptures as rpt
import matplotlib.pyplot as plt

import init_path
import shutil
from orion.utils.misc_utils import *
from orion.algos.tap_segmentation import TAPSegmentation
from orion.utils.gpt4v_utils import encode_imgs_from_path, json_parser
from orion.algos.gpt4v import GPT4V

def calculate_waypoints_prompt():
    prompt = '''You are a humanoid robot. Assume you have oracles for grasping objects, placing objects at certain locations and pushing objects to certain positions. 
You goal is to finish a task by imitating human. 
Here I give you images of a human finishing a task. You need to determine if you need to imitate whole human trajectory, or if you can just centering on the target pose given your oracles. 
You need to choose from 'Whole Trajectory' or 'Target Only'. Your output format is:

```json
{
    "Answer": YOUR_CHOICE('Target Only' or 'Whole Trajectory'),
}
Reasons: YOUR_REASONS
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc. 
Note that you shouldn't output 'Whole Trajectory' unless you are absolutely sure. Don't make any guesses, and only focus on the motion shown in images!
'''
    return prompt

def calculate_waypoints(vlm, img_paths):
    vlm.begin_new_dialog()
    base64_img_list = [encode_imgs_from_path(vlm, img_paths)]
    text_prompt_list = [calculate_waypoints_prompt()]
    text_response = vlm.run(text_prompt_list, base64_img_list)
    print(text_response)

    json_data = json_parser(text_response)
    if json_data is None:
        return None
    return json_data

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument('--num-waypoints', type=int, default=5, help='Number of waypoints to calculate.')
    args = parser.parse_args()

    tap_segmentation = TAPSegmentation()
    tap_segmentation.load(args.annotation_folder)
    print("num_segments:", len(tap_segmentation.temporal_segments.segments))

    video_seq = get_video_seq_from_annotation(args.annotation_folder)
    waypoints_indices = []
    for seg in tap_segmentation.temporal_segments.segments:
        waypoints_indices.append(np.linspace(seg.start_idx, seg.end_idx, args.num_waypoints, dtype=int))

    tmp_path = "tmp_images"
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    vlm = GPT4V()
    res = []
    for i in range(len(waypoints_indices)):
        
        img_paths = []
        for j in range(len(waypoints_indices[i])):
            keyframe = video_seq[waypoints_indices[i][j]]
            keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(os.path.join(tmp_path, "images", f"frame_{j}.jpg")), keyframe)
            img_paths.append(os.path.join(os.path.join(tmp_path, "images", f"frame_{j}.jpg")))
        
        json_data = calculate_waypoints(vlm, img_paths)
        # print("results=", json_data["Answer"])
        res.append(json_data["Answer"])

    # remove the folder
    shutil.rmtree(tmp_path)

    with open(os.path.join(args.annotation_folder, "waypoints_info.json"), "w") as f:
        json.dump(res, f)

if __name__ == "__main__":
    main()