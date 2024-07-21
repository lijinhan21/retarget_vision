import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument("--no-depth", action="store_true", default=False)
    args = parser.parse_args()

    print("*************Run HaMer*************")
    commands = [
        "python",
        "scripts_new/06a_hand_analysis.py",
        "--annotation-folder",
        args.annotation_folder,
    ]
    command = " ".join(commands)
    os.system(command)

    print("*****Analyze Hand Object Contact*****")
    commands = [
        "python",
        "scripts_new/06b_hand_object_contact_calculation.py",
        "--annotation-folder",
        args.annotation_folder,
        "--no-depth" if args.no_depth else "",
    ]
    command = " ".join(commands)
    os.system(command)


if __name__ == '__main__':
    main()