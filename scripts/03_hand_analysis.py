from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import h5py
import shutil
import imageio

import init_path
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def get_annotation_info(annotations_folder):
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    with open(os.path.join(annotations_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
    return config_dict

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--render-left', action='store_true', default=False, help='Render right hand instead of left hand')

    parser.add_argument('--annotation-path', type=str, required=True, help='Path to human demo annotation file file')
    parser.add_argument('--flip-lr', action='store_true')
    parser.add_argument('--flip-rgb', action='store_true')
    args = parser.parse_args()

    assert(os.path.exists(args.annotation_path)), f"Annotation file {args.annotation_path} does not exist!"

    config_dict = get_annotation_info(args.annotation_path)

    demo_file_path = config_dict["original_file"]
    temp_img_dir = os.path.join(args.annotation_path, "temp_input")
    out_folder = os.path.join(args.annotation_path, "hamer_output")
    os.makedirs(temp_img_dir, exist_ok=True)
    with h5py.File(demo_file_path, "r") as f:
        images = f["data/human_demo/obs/agentview_rgb"][()]
        if args.flip_lr:
            flip_lr = -1
        else:
            flip_lr = 1

        if args.flip_rgb:
            flip_rgb = -1
        else:
            flip_rgb = 1
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(temp_img_dir, f"{i:04d}.jpg"), image[:, ::flip_lr, ::flip_rgb])

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in sorted(Path(temp_img_dir).glob(end))]

    # Iterate over all images in folder
    hand_joints_seq = {}
    hand_keypoints_seq = {}
    bbox_size_seq = {}
    bbox_center_seq = {}
    verts_seq = []
    person_ids = []
    vitpose_detections = []

    num_total_frames = len(img_paths)

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            # keyp = left_hand_keyp
            # valid = keyp[:,2] > 0.5
            # if sum(valid) > 3:
            #     bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            #     bboxes.append(bbox)
            #     is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        vitpose_detections.append(right_hand_keyp)
        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints_3d = {}
        all_joints_2d = {}

        all_bboxes_center = {}
        all_bboxes_size = {}

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                person_ids.append(person_id)
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(os.path.join(out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                # verts: (778, 3)
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                joints_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                joints_2d = out["pred_keypoints_2d"][n].detach().cpu().numpy()

                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                if person_id not in all_joints_3d:
                    all_joints_3d[person_id] = []
                    all_joints_2d[person_id] = []

                torch.save({"joints_2d": joints_2d, "joints_3d": joints_3d, "bbox_center": box_center.detach().cpu().numpy(), "bbox_size": box_size.detach().cpu().numpy()}, os.path.join(out_folder, f'{img_fn}_{person_id}_joints.pt'))
                verts_path = str(img_path).replace(".jpg", "_verts.pt")
                torch.save({"verts": verts}, verts_path)
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(out_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view

        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )

            for i in range(len(all_verts)):
                if all_right[i] is not args.render_left:
                    cam_view = renderer.render_rgba_multiple(all_verts[i:i+1], cam_t=all_cam_t[i:i+1], render_res=img_size[n], is_right=all_right[i:i+1], **misc_args)

                    tip_verts = np.stack([
                        all_verts[i][744], # thumb
                        all_verts[i][320], # index
                        all_verts[i][443], # middle
                        all_verts[i][554], # ring
                        all_verts[i][671], # pinky
                    ])
                    break

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    for key in np.unique(person_ids):
        for i in range(num_total_frames):
            file = os.path.join(out_folder, f"{i:04d}_{key}_joints.pt")
            if os.path.exists(file):
                data = torch.load(file)
            else:
                data = {
                    "joints_3d": np.zeros((21, 3)),
                    "joints_2d": np.zeros((21, 2)),
                }
            if key not in hand_joints_seq:
                hand_joints_seq[key] = []
                hand_keypoints_seq[key] = []
                bbox_size_seq[key] = []
                bbox_center_seq[key] = []
            hand_joints_seq[key].append(data["joints_3d"])
            hand_keypoints_seq[key].append(data["joints_2d"])
            # bbox_size_seq[key].append(data["bbox_size"])
            # bbox_center_seq[key].append(data["bbox_center"])
    for key in hand_joints_seq.keys():
        hand_joints_seq[key] = np.stack(hand_joints_seq[key], axis=0)
        hand_keypoints_seq[key] = np.stack(hand_keypoints_seq[key], axis=0)
        # bbox_size_seq[key] = np.stack(bbox_size_seq[key], axis=0)
        # bbox_center_seq[key] = np.stack(bbox_center_seq[key], axis=0)

    if len(hand_joints_seq.keys()) == 0:
        print("No hands detected!")
    else:
        torch.save({"hand_joints_seq": hand_joints_seq, "hand_keypoints_seq": hand_keypoints_seq,  "vitpose_detections": vitpose_detections}, os.path.join(out_folder, f'hand_keypoints.pt'))

    shutil.rmtree(temp_img_dir)

if __name__ == '__main__':
    main()
