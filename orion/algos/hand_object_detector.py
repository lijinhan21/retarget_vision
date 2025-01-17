# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

class HandObjectDetector:
    def __init__(self):
        cfg_from_file('configs/hand_object_detect/res101.yml')
        cfg.USE_GPU_NMS = True
        np.random.seed(cfg.RNG_SEED)

        # load model
        model_dir = "third_party/hand_object_detector/models/res101_handobj_100K/pascal_voc"
        load_name = os.path.join(model_dir, 'faster_rcnn_1_8_132028.pth')
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        cfg.cuda = True
        self.fasterRCNN.cuda()
        print('load model successfully!')

    def detect(self, im_file, thresh_hand=0.5, thresh_obj=0.5, vis=True, save_path='./dets'):
        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1).cuda()
        im_info = torch.FloatTensor(1).cuda()
        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()
        box_info = torch.FloatTensor(1).cuda()

        with torch.no_grad():
            self.fasterRCNN.eval()

            im = cv2.imread(im_file)
            print("img=", im_file)

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_() 
            
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0] # hand contact state info
            offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach() # hand side info (left/right)

            # get hand contact 
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side 
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            
            pred_boxes /= im_scales[0]
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            if vis:
                im2show = np.copy(im)

            obj_dets, hand_dets = None, None
            for j in xrange(1, len(self.pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if self.pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
                elif self.pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores, 0, True)

                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if self.pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if self.pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
                
            if vis:
                # visualization
                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj, font_path='third_party/hand_object_detector/lib/model/utils/times_b.ttf')
                # print("hand_dets: ", hand_dets)

                folder_name = save_path
                os.makedirs(folder_name, exist_ok=True)
                im_file_name = im_file.split('/')[-1][:-4]
                result_path = os.path.join(folder_name, im_file_name + "_det.png")
                im2show.save(result_path)
            
            return obj_dets, hand_dets
        
    def parse_hand_info(self, hand_dets):
        
        hand_res = {
            'left': {'in_contact': False, 'contact_type': 'none'},
            'right': {'in_contact': False, 'contact_type': 'none'}
        }
        if hand_dets is None:
            return hand_res
        
        for det in hand_dets:
            if det[-1] == 0: # left hand
                hand_res['left']['in_contact'] = int(det[5]) != 0
                hand_res['left']['contact_type'] = 'portable' if int(det[5]) == 3 else ('stationary' if int(det[5]) == 4 else 'none')
            elif det[-1] == 1: # right hand
                hand_res['right']['in_contact'] = int(det[5]) != 0
                hand_res['right']['contact_type'] = 'portable' if int(det[5]) == 3 else ('stationary' if int(det[5]) == 4 else 'none')
        
        return hand_res

if __name__ == '__main__':

    detector = HandObjectDetector()
    obj_dets, hand_dets = detector.detect('third_party/hand_object_detector/images/frame_054.jpg', vis=True, save_path='third_party/hand_object_detector/images_dets/')
    print(detector.parse_hand_info(hand_dets))