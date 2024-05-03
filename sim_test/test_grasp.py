import argparse
import pickle
import time
import json
import tabulate
import os

import cv2
import numpy as np
import math

import mujoco
from mujoco import viewer

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import xml.etree.ElementTree as ET

import init_path
from orion.utils.misc_utils import *
from orion.utils.robosuite_utils import *
from orion.algos.hoig import HandObjectInteractionGraph
from orion.algos.human_video_hoig import HumanVideoHOIG
from orion.algos.grasp_primitives import GraspPrimitive

def get_action(args, step, obs, grasp_dict, traj, **kwargs):
    
    target_joint_pos = np.zeros(32)

    init_interpolate_steps = 30
    if step < init_interpolate_steps:
        # interpolate to traj[0]
        ed = calculate_target_qpos(traj[0])
        st = np.array([
            -0.0035970834452681436, 0.011031227286351492, -0.01311470003464996, 0.0, 0.0, 0.0, 0.8511509067155127, 1.310805039853726, -0.7118440391862395, -0.536551596928798, 0.02341464067352966, -0.23317144423063796, -0.0803808564555934, 0.18086797377837605, -1.5034221574091646, -0.15101789788918812, 0.00014316406250000944, -0.07930486850248092, -0.1222325540688668, -0.2801763429367678,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ])
        # st = np.array([0.] * 32)
        # st[6] = -2
        # st[9] = -0.8
        target_joint_pos = st + (ed - st) * (step / init_interpolate_steps)
    else:
        if step == init_interpolate_steps:
            print("init interpolation finished. replay begin!")
        # head, torso, and arms
        target_joint_pos = calculate_target_qpos(traj[min(step - init_interpolate_steps, len(traj) - 1)])

        hand_primitive_l = grasp_dict.point_map_to_primitive(target_joint_pos[-6:])
        hand_primitive_r = grasp_dict.point_map_to_primitive(target_joint_pos[-12:-6])
        target_joint_pos[-6:] = hand_primitive_l[0]
        target_joint_pos[-12:-6] = hand_primitive_r[0]

        if step < 140:
            target_joint_pos[-12:-6] = grasp_dict.get_joint_angles("palm")
        print("r primitive name=", hand_primitive_r[1])

        # mean_error = np.mean(np.abs(target_joint_pos[:20] - obs["robot0_joint_pos"]))
        # print("mean error:", mean_error)

    action = calculate_action_from_target_joint_pos(obs, target_joint_pos)
    return action

def get_action_with_grasp_primitive(args, step, obs, grasp_dict, traj, **kwargs):
    
    target_joint_pos = np.zeros(32)

    init_interpolate_steps = 30
    if step < init_interpolate_steps:
        # interpolate to traj[0]
        ed = calculate_target_qpos(traj[0])
        st = np.array([
            -0.0035970834452681436, 0.011031227286351492, -0.01311470003464996, 0.0, 0.0, 0.0, 0.8511509067155127, 1.310805039853726, -0.7118440391862395, -0.536551596928798, 0.02341464067352966, -0.23317144423063796, -0.0803808564555934, 0.18086797377837605, -1.5034221574091646, -0.15101789788918812, 0.00014316406250000944, -0.07930486850248092, -0.1222325540688668, -0.2801763429367678,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ])
        # st = np.array([0.] * 32)
        # st[6] = -2
        # st[9] = -0.8
        target_joint_pos = st + (ed - st) * (step / init_interpolate_steps)
    else:
        if step == init_interpolate_steps:
            print("init interpolation finished. replay begin!")
        # head, torso, and arms
        target_joint_pos = calculate_target_qpos(traj[min(step - init_interpolate_steps, len(traj) - 1)])

        # mean_error = np.mean(np.abs(target_joint_pos[:20] - obs["robot0_joint_pos"]))
        # print("mean error:", mean_error)

    action = calculate_action_from_target_joint_pos(obs, target_joint_pos)
    return action

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument("--environment", type=str, default="HumanoidSimple")
    parser.add_argument("--save-path", type=str, default="sim_test/output.mp4")
    args = parser.parse_args()

    human_video_hoig = HumanVideoHOIG()
    human_video_hoig.generate_from_human_video(args.annotation_folder, 1.2, use_smplh=True)

    offset = [-0., -0.0, 0.0]
    # whole_traj, ik_traj = human_video_hoig.get_retargeted_ik_traj() # offset={"link_RArm7": offset, "link_LArm7": offset}
    # print(whole_traj.shape)
    # for i in ik_traj:
    #     print(i.shape)
    
    whole_traj, ik_traj = human_video_hoig.get_retargeted_ik_traj_with_grasp_primitive(offset={"link_RArm7": offset, "link_LArm7": offset}) # offset={"link_RArm7": offset, "link_LArm7": offset}
    print(whole_traj.shape)
    for i in ik_traj:
        print(i.shape)

    # save trajectory
    with open(os.path.join(args.annotation_folder, 'ik_traj_with_grasp_primitives.npy'), "wb") as f:
        np.save(f, whole_traj)

    grasp_primitive = GraspPrimitive()
    launch_simulator(args, 
                     get_action_func=get_action_with_grasp_primitive, 
                     max_steps=500, 
                     offset=offset, 
                     grasp_dict=grasp_primitive, 
                     traj=whole_traj)

if __name__ == "__main__":
    main()