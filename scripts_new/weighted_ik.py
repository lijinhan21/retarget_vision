import argparse
import pickle
import time
import json
import tabulate
import os

import cv2
import numpy as np

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
from retarget.utils.constants import name_to_urdf_idx

def stitch_video(data, offset, video_idx):
    video_paths = [f'tmp_video/{i}.mp4' for i in range(8)]
    video_clips = [cv2.VideoCapture(path) for path in video_paths]
    width = int(video_clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"tmp_video/mouse_stitched_{video_idx}.mp4", fourcc, 3, (width * 2, height))

    out_frames = []
    while True:
        frames = [cap.read()[1] for cap in video_clips]  # read a frame from each video
        if any(frame is None for frame in frames):
            break  # stop if any video reaches the end
        
        # cut frame size
        frames = [frame[270:-270, 480:-480] for frame in frames]

        # add text on each frame
        for i, frame in enumerate(frames):
            w_offset = 50
            h_offset = 50
            cv2.putText(frame, f"shoulder", (10 + w_offset, 50 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][0] > 0.5)), 2) # {data[-1][i]:.2f}
            cv2.putText(frame, f"elbow", (10 + w_offset, 100 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][2] > 0.5)), 2)
            cv2.putText(frame, f"wrist", (10 + w_offset, 150 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][4] > 0.5)), 2)

            cv2.putText(frame, f"w", (170 + w_offset, 20 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"{data[i][0]:.1f}", (150 + w_offset, 50 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][0] > 0.5)), 2) 
            cv2.putText(frame, f"{data[i][2]:.1f}", (150 + w_offset, 100 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][2] > 0.5)), 2)
            cv2.putText(frame, f"{data[i][4]:.1f}", (150 + w_offset, 150 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][4] > 0.5)), 2)  

            cv2.putText(frame, f"err", (270 + w_offset, 20 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"{data[i][1]:.2f}", (250 + w_offset, 50 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][0] > 0.5)), 2) 
            cv2.putText(frame, f"{data[i][3]:.2f}", (250 + w_offset, 100 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][2] > 0.5)), 2)
            cv2.putText(frame, f"{data[i][5]:.2f}", (250 + w_offset, 150 + h_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225 * int(data[i][4] > 0.5)), 2)  

        # Assume we're doing a 2x4 grid as rows of 4 videos each
        row1 = np.hstack((frames[0], frames[1], frames[2], frames[3]))
        row2 = np.hstack((frames[4], frames[5], frames[6], frames[7]))

        # Vertically stack the rows
        final_frame = np.vstack((row1, row2))

        # add offset text
        cv2.putText(final_frame, f"offset={offset}", (1600, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the stitched frame
        out.write(final_frame)
        out_frames.append(final_frame)
    
    # repead out_frames for 10 times
    for _ in range(9):
        for frame in out_frames:
            out.write(frame)

    # Release everything when done
    for cap in video_clips:
        cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Visualize recorded data from smpl")
    parser.add_argument("--input", type=str, help="data to streaming results")
    parser.add_argument(
        "--config", type=str, help="data to streaming results", default="GR1_retarget/configs/smpl_gr1.yaml"
    )
    parser.add_argument("--save-video", action="store_true", default=False)
    args = parser.parse_args()
    config = load_config(args.config)
    # s = np.load(args.input).astype(np.float64)  # T 52 4 4

    retargeter = SMPLGR1Retargeter(config, vis=True)

    with open(args.input, "rb") as f:
        s = pickle.load(f)
    # s = np.load(args.input)

    print("data len=", len(s))
    s = s[90:]
    print("new data len=", len(s))

    data0 = s[0]

    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")

    with open("scripts_new/ik_weight_template.json", "r") as f:
        ik_weights = json.load(f)

    exp_idx = 0
    exp_weights = [
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        # [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        # [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        # [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ]
    data = []
    headers = ["shoulder weight", "shoulder err", "elbow weight", "elbow err", "wrist weight", "wrist err"]

    offsets = [{
        "link_RArm7": [0.0, 0.0, 0.0],
    },
    # {
    #     "link_RArm7": [0.0, 0.1, 0.0],
    # },{
    #     "link_RArm7": [0.0, -0.1, 0.0],
    # },{
    #     "link_RArm7": [0.0, 0.0, -0.15],
    # },
    {
        "link_RArm7": [0.0, 0.0, 0.15],
    },{
        "link_RArm7": [0.0, 0.1, -0.15],
    },{
        "link_RArm7": [0.0, -0.1, 0.15],
    },
    ]

    for offset_idx, offset in enumerate(offsets):
        exp_idx = 0
        data = []
        while True:
            # break
            if exp_idx >= len(exp_weights):
                break
            
            if args.save_video:
                # 1080p
                os.makedirs('tmp_video', exist_ok=True)
                writer = cv2.VideoWriter(f'tmp_video/{exp_idx}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20, (1920, 1080))
                time.sleep(1)

            for idx, link in enumerate(ik_weights["GR1_body"]):
                ik_weights["GR1_body"][link]["position_cost"] = exp_weights[exp_idx][idx]
            retargeter.update_weights(ik_weights)
            total_error = {}
            exp_idx += 1

            for data_t in s:
                _, error, __ = retargeter(data_t, offset)
                for k in error.keys():
                    if k not in total_error:
                        total_error[k] = 0
                    total_error[k] += error[k]
                # break
                if args.save_video:
                    rgba = retargeter.vis.viz.captureImage()[:, :, :3]
                    # make the size 1080p, also drop the a channel
                    rgba = cv2.resize(rgba, (1920, 1080))
                    rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    writer.write(rgba)
                time.sleep(0.05)
            
            # print("ik weights=", ik_weights["GR1_body"])
            for k in total_error.keys():
                total_error[k] /= len(s)
            # print("Total error:", total_error)
            data.append([
                ik_weights["GR1_body"]['link_RArm2']["position_cost"],
                total_error["link_RArm2"],
                ik_weights["GR1_body"]['link_RArm4']["position_cost"],
                total_error["link_RArm4"],
                ik_weights["GR1_body"]['link_RArm7']["position_cost"],
                total_error["link_RArm7"],
            ])
            print("exp_idx=", exp_idx, "finished")

            if args.save_video:
                writer.release()
                print(f"Saved video to tmp_video/{exp_idx}.mp4")
                #break
            
        print(tabulate.tabulate(data, headers=headers))
        
        # combine all videos
        if args.save_video:
            stitch_video(data, offset, offset_idx)



if __name__ == "__main__":
    main()
