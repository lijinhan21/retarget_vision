import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import yaml
import argparse
from orion.algos.human_video_oogs import HumanVideoOOGs
from orion.algos.oog import OpenWorldObjectSceneGraph
from orion.utils.misc_utils import *
from orion.utils.o3d_utils import *
from orion.utils.correspondence_utils import CorrespondenceModel
from orion.utils.real_robot_utils import ImageCapturer
from orion.utils.log_utils import get_orion_logger

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.config_utils import robot_config_parse_args

from deoxys_vision.networking.camera_redis_interface import \
    CameraRedisSubInterface
from deoxys_vision.utils.calibration_utils import load_default_extrinsics, load_default_intrinsics
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.utils.img_utils import save_depth_in_rgb


from real_robot_scripts.ik_execution import ik_execution
from real_robot_scripts.rollout_tracking import RolloutTracker

logger = get_orion_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", type=str)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    robot_config_parse_args(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    is_finished = False
    image_capturer = ImageCapturer(record=not args.no_record)
    human_video_oogs = HumanVideoOOGs()

    correspondence_model = CorrespondenceModel()
    human_video_oogs.set_correspondence_model(correspondence_model)
    

    deviation_angle_threshold = 30
    skip_trivial_solution = True
    high_occlusion = False
    offset = [0.0, -0.02, 0.03]     # system offset due to calibration errors. This offset needs to be adjusted for individual system, but you should only adjust it only once. 
    velocity_threshold = 1.0
    target_dist_threshold = 0.01
    target_intersection_threshold = 100

    runtime_folder, rollout_folder, human_video_annotation_path = read_from_runtime_file()

    robot_video_annotation_path = rollout_folder

    human_video_oogs.generate_from_human_video(human_video_annotation_path)
    human_video_oogs.plan_inference(velocity_threshold=velocity_threshold,
                                    target_dist_threshold=target_dist_threshold, 
                                    target_intersection_threshold=target_intersection_threshold)

    # Algorithm starts
    rollout_tracker = RolloutTracker()
    while not is_finished:
        tmp_annotation_path = os.path.join(runtime_folder, "tmp_annotation.png")

        # 1. check OOG matching
        is_first_frame = True
        if os.path.exists(tmp_annotation_path):
            is_first_frame = False

        robot_object_graph = OpenWorldObjectSceneGraph()
        if is_first_frame:
            robot_depth_path = os.path.join(robot_video_annotation_path, "depth.png") # "traj_exp/testing/assembly_plane_testing_2_depth.png"
            robot_first_frame, robot_annotation = get_first_frame_annotation(robot_video_annotation_path)

            last_obs = image_capturer.get_last_obs()
            robot_object_graph.generate_from_robot_demo(
                input_image=robot_first_frame,
                input_depth=load_depth_in_rgb(robot_depth_path),
                camera_intrinsics=last_obs["intrinsics"],
                camera_extrinsics=last_obs["extrinsics"],
                # Comment this line, and the model will first use SAM to get a glboal segmentation mask
                input_annotation=robot_annotation,
                reference_graph=human_video_oogs.get_graph(0),
                is_first_frame=is_first_frame,
                correspondence_model=human_video_oogs.correspondence_model,
            )
            update_annotation = Image.fromarray(robot_object_graph.input_annotation)
            update_annotation.putpalette(get_palette())
            update_annotation.save(os.path.join(robot_video_annotation_path, "frame_annotation.png"))

        else:
            logger.info("loading from previous step")
            last_obs = image_capturer.get_last_obs()
            current_image, current_depth = last_obs["color_img"], last_obs["depth_img"]
            # robot_depth_path = os.path.join(robot_video_annotation_path, "depth.png") # "traj_exp/testing/assembly_plane_testing_2_depth.png"

            tmp_image_path = os.path.join(runtime_folder, "tmp.jpg")

            robot_annotation = np.array(Image.open(tmp_annotation_path))
            # dataset_name = get_dataset_name_from_annotation(reference_path)
            # info = load_reconstruction_info_from_human_demo(dataset_name)

            robot_object_graph.generate_from_robot_demo(
                input_image=current_image[..., ::-1],
                input_depth=current_depth,
                camera_intrinsics=last_obs["intrinsics"],
                camera_extrinsics=last_obs["extrinsics"],
                # Comment this line, and the model will first use SAM to get a glboal segmentation mask
                input_annotation=robot_annotation,
                reference_graph=human_video_oogs.get_graph(0),
                is_first_frame=is_first_frame,
                correspondence_model=human_video_oogs.correspondence_model,
            )

        # human_video_oogs.get_graph(0).draw_overlay_image(mode="all")
        robot_object_graph.draw_overlay_image(mode="object")
        # human_video_oogs.get_graph(0).draw_dense_correspondence(robot_object_graph, object_ids=[2])

        robot_object_graph.compute_contact_states(
            # dist_threshold=0.005, 
            # intersection_threshold=50
            )
        # matched_graph_idx = 3
        print(human_video_oogs.get_oog_mode_sequence())
        matched_graph_idx = human_video_oogs.find_matching_oog_idx(robot_object_graph)
        current_oog_idx = matched_graph_idx

        manipulate_object_id = human_video_oogs.get_manipulate_object_seq()[matched_graph_idx]
        reference_object_id = human_video_oogs.get_reference_object_seq()[matched_graph_idx]
        logger.debug(f"manipulate id: {manipulate_object_id}")
        logger.debug("reference id: {reference_object_id}")
        
        logger.info(f"Mathced graph idx is: {matched_graph_idx}")
        logger.debug(f"Robot current contact state is: {robot_object_graph.contact_states}")

        if matched_graph_idx >= human_video_oogs.num_graphs - 1:
            logger.info("Finished!")
            is_finished = True
            continue
        # End of checking OOG matching

        # 2. Compute ik from observation
        subgoal_transform, target_transform = human_video_oogs.compute_subgoal(
            matched_idx=matched_graph_idx,
            robot_object_graph=robot_object_graph,
            manipulate_object_id=manipulate_object_id,
            reference_object_id=reference_object_id,
            high_occlusion=high_occlusion,
            deviation_angle_threshold=deviation_angle_threshold,
            skip_trivial_solution=skip_trivial_solution
        )

        R_seq, t_seq,  best_loss, training_data = robot_object_graph.estimate_motion_traj_from_object(
            object_id=manipulate_object_id, 
            use_visibility=True,
            skip_interval=1,
            # select_subset=3,
            mode="lie",
            regularization_weight_pos=0.1,
            regularization_weight_rot=0.1,    
            num_max_iter=3,
            high_occlusion=high_occlusion,
            optim_kwargs={
                "lr": 0.01,
                "num_epochs": 501,
                "verbose": True,
                "momentum": 0.9,
            }
        )

        logger.debug("target transforma: ", target_transform)

        object_1_pcd_points, object_1_pcd_colors = robot_object_graph.get_objects_3d_points(object_id=manipulate_object_id, remove_outlier_kwargs={"nb_neighbors": 30, "std_ratio": 0.7})
        object_2_pcd_points, object_2_pcd_colors = robot_object_graph.get_objects_3d_points(object_id=reference_object_id)
        demo_object_1_pcd_points, demo_object_1_pcd_colors = human_video_oogs.get_graph(current_oog_idx).get_objects_3d_points(object_id=manipulate_object_id)
        demo_object_2_pcd_points, demo_object_2_pcd_colors = human_video_oogs.get_graph(current_oog_idx).get_objects_3d_points(object_id=reference_object_id)

        logger.debug(f"t_seq information: {t_seq.shape}, sum: {np.sum(t_seq, axis=0)}")

        new_R = np.eye(3)
        new_R_seq = []
        R_seq = R_seq
        for R in R_seq:
            new_R = R @ new_R
            new_R_seq.append(new_R)
        logger.debug(new_R_seq[-1])

        new_points = []
        new_point = object_1_pcd_points
        for R, t in zip(R_seq, t_seq):
            new_point = (R @ (new_point - new_point.mean(axis=0)).T ).T + new_point.mean(axis=0) + t[None, :]
            # new_point = new_point + t[None, :]
            new_points.append(new_point)

        def transform_point_clouds(transformation, points):
            new_points = transformation @ np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
            new_points = new_points[:3, :].T
            return new_points

        subgoal_points = human_video_oogs.get_graph(current_oog_idx).get_world_trajs(object_ids=[manipulate_object_id])[:, -1, :]
        subgoal_points = transform_point_clouds(subgoal_transform, subgoal_points)

        object_1_new_points = new_points[-1]

        manipulation_offset = object_1_pcd_points.mean(axis=0) - demo_object_1_pcd_points.mean(axis=0)
        reference_offset = object_2_pcd_points.mean(axis=0) - demo_object_2_pcd_points.mean(axis=0)
        new_demo_object_1_pcd_points = transform_point_clouds(target_transform, demo_object_1_pcd_points + manipulation_offset)

        eef_node = human_video_oogs.get_graph(current_oog_idx).eef_node
        gripper_action = eef_node.get_eef_action(verbose=True)

        interaction_points = eef_node.interaction_affordance.get_interaction_points(include_centroid=False)
        interaction_points = transform_point_clouds(target_transform, interaction_points + manipulation_offset)

        interaction_offset = np.array(offset).reshape(1, 3)
        interaction_centroid = eef_node.interaction_affordance.get_affordance_centroid()
        interaction_centroid = transform_point_clouds(target_transform, interaction_centroid + manipulation_offset)
        interaction_centroid += interaction_offset


        z_rotation = np.arctan2(target_transform[1, 0], target_transform[0, 0])

        logger.debug(f"object_1_new_points mean: {object_1_new_points.mean(axis=0)}, object_1_pcd_points mean: {object_1_pcd_points.mean(axis=0)}")
        logger.debug("z rotation: ", z_rotation)

        new_demo_object_2_pcd_points = transform_point_clouds(subgoal_transform, demo_object_2_pcd_points + reference_offset)

        runtime_T_file = os.path.join(robot_video_annotation_path, f"T_{matched_graph_idx}_seq.pt")
        torch.save({"R_seq": R_seq, "t_seq": t_seq, 
                    "target_object_centroid": object_1_pcd_points.mean(axis=0),
                    "target_interaction_centroid": interaction_centroid,
                    "target_interaction_points": interaction_points,
                    "z_rotation": z_rotation
                    }, runtime_T_file)

        with open("experiments/runtime_T.json", "w") as f:
            json.dump({"file": runtime_T_file}, f)
        # Execute motion trajectories
        ik_execution(args, image_capturer)
        # annotate rollout video
        rollout_tracker.run()



if __name__ == "__main__":
    main()