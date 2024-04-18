import os
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default=None)
    parser.add_argument("--tracker-type", type=str, default="cutie", choices=["xmem", "cutie"])
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument('--tap-pen', type=float, default=10, help='Penalty for changepoint detection.')
    args = parser.parse_args()

    # 1. xmem segmentation
    print("*************XMem Segmentation*************")
    commands = [
        "python",
        "scripts/02a_xmem_annotation.py",
        "--annotation-folder",
        args.annotation_folder,
        "--tracker-type",
        args.tracker_type,
    ]
    command = " ".join(commands)
    os.system(command)

    # 2. cotracker annotation
    print("*************Cotracker Annotation*************")
    commands = [
        "python",
        "scripts/02b_generate_cotracker_annotation.py",
        "--annotation-folder",
        args.annotation_folder,
        "--num-track-points",
        "40",
    ]
    if not args.save_video:
        commands.append("--no-video")
    command = " ".join(commands)
    os.system(command)
    
    # 3. tap-based temporal segmentation
    print("*************Tap Annotation*************")
    commands = [
        "python",
        "scripts/02c_pt_changepoint_segmentation.py",
        "--annotation-folder",
        args.annotation_folder,
        "--pen",
        str(args.tap_pen),
    ]
    command = " ".join(commands)
    os.system(command)

    # # 3. hamer annotation
    # print("*************Hamer Annotation*************")
    # commands = [
    #     "python",
    #     "scripts/generate_hamer_annotation.py",

if __name__ == '__main__':
    main()