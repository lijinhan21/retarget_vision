import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_demo", type=str, default=None)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument('--tap-pen', type=float, default=10, help='Penalty for changepoint detection.')
    args = parser.parse_args()

    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    # 1. generate text description
    # print("*************Text Description*************")
    # commands = [
    #     "python",
    #     "scripts_new/01_generate_descriptions.py",
    #     "--human_demo",
    #     args.human_demo,
    # ]
    # command = " ".join(commands)
    # os.system(command)
    
    # 2. gam annotation
    print("*************GAM Annotation*************")
    commands = [
        "python",
        "scripts_new/02_gam_annotation.py",
        "--human_demo",
        args.human_demo,
    ]
    command = " ".join(commands)
    os.system(command)

    # 3. xmem segmentation
    print("*************XMem Segmentation*************")
    commands = [
        "python",
        "scripts_new/03_xmem_annotation.py",
        "--annotation-folder",
        annotation_path
    ]
    command = " ".join(commands)
    os.system(command)

    # 4. cotracker annotation
    print("*************Cotracker Annotation*************")
    commands = [
        "python",
        "scripts_new/04_generate_cotracker_annotation.py",
        "--annotation-folder",
        annotation_path,
        "--num-track-points",
        "40",
    ]
    if not args.save_video:
        commands.append("--no-video")
    command = " ".join(commands)
    os.system(command)
    
    # 5. tap-based temporal segmentation
    print("*************Tap Annotation*************")
    commands = [
        "python",
        "scripts_new/05_pt_changepoint_segmentation.py",
        "--annotation-folder",
        annotation_path,
        "--pen",
        str(args.tap_pen),
    ]
    command = " ".join(commands)
    os.system(command)

if __name__ == '__main__':
    main()