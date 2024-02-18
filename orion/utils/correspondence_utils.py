import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from collections import Counter

from orion.algos.sam_operator import SAMOperator
from orion.utils.misc_utils import get_first_frame_annotation, overlay_xmem_mask_on_image, resize_image_to_same_shape, plotly_draw_seg_image
from orion.algos.dino_features import DinoV2ImageProcessor, compute_affinity, rescale_feature_map, generate_video_from_affinity


def compute_iou(reference_mask, segmentation_masks):
    # Convert to binary arrays
    reference_mask = reference_mask.astype(bool)
    segmentation_masks = segmentation_masks.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(reference_mask, segmentation_masks)
    union = np.logical_or(reference_mask, segmentation_masks)

    # Compute IoU
    iou = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
    return iou

def find_most_repeated_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else -1

class CorrespondenceModel(nn.Module):
    def __init__(self,
                 dinov2: DinoV2ImageProcessor = None,
                 sam: SAMOperator = None,
                 reference_size=(448, 448)):
        super().__init__()

        self.dinov2 = DinoV2ImageProcessor() if dinov2 is None else dinov2
        self.sam_operator = SAMOperator() if sam is None else sam
        self.sam_operator.init()

        self.reference_size = reference_size

    def init(self):
        self.sam_operator.init()

    def segment_image(self, image):
        results = self.sam_operator.segment_image(image)
        return results["merged_mask"]

    def compute_cost_volume(self,
                            current_obs_image,
                            ref_image,
                            h=32,
                            w=32):
        """_summary_

        Args:
            current_obs_image (np.ndarray): _description_
            ref_image (np.ndarray): _description_
            h (int, optional): height of the feature map. Defaults to 32.
            w (int, optional): width of the feature map. Defaults to 32.

        Returns:
            torch.tensor: cost volume of DINO features.
        """
        current_obs_image = resize_image_to_same_shape(current_obs_image, ref_image)
        img_list = []
        feature_list = []
        for img in [ref_image, current_obs_image]:
            img_list.append(img)
            feature_list.append(self.dinov2.process_image(img))
        aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w))
        return aff

    def get_patchified_mask(self, resized_instance_mask, patch_size):
        patchified_mask = rearrange(resized_instance_mask, '(h p1) (w p2) -> h w (p1 p2)', p1=patch_size, p2=patch_size)
        patchified_mask = np.sum(patchified_mask, axis=-1) / (patch_size * patch_size)
        return patchified_mask

    def object_correspondence(self, 
                              current_obs_image_input, 
                              current_annotation_mask_input,
                              ref_image_input, 
                              ref_annotation_mask_input,
                              h=32,
                              w=32,
                              patch_size=14,
                              threshold=0.3,
                              temperature=100,
                              topk=30):
        """_summary_

        Args:
            current_obs_image_input (_type_): _description_
            current_annotation_mask_input (_type_): _description_
            ref_image_input (_type_): _description_
            ref_annotation_mask_input (_type_): _description_
            h (int, optional): _description_. Defaults to 32.
            w (int, optional): _description_. Defaults to 32.
            patch_size (int, optional): _description_. Defaults to 14.
            threshold (float, optional): _description_. Defaults to 0.5.
            temperature (int, optional): _description_. Defaults to 100.
            topk (int, optional): _description_. Defaults to 10.

        Returns:
            np.ndarray: the new annotation
        """
        
        # 1. Reshape segmetnation mask.
        current_obs_image = resize_image_to_same_shape(current_obs_image_input, reference_size=self.reference_size)
        current_annotation_mask = resize_image_to_same_shape(current_annotation_mask_input, reference_size=self.reference_size)
        ref_image = resize_image_to_same_shape(ref_image_input, reference_size=self.reference_size)
        ref_annotation_mask = resize_image_to_same_shape(ref_annotation_mask_input, reference_size=self.reference_size)

        # 2. Split a single mask into individual for better processing.
        current_masks = []
        for mask_id in range(current_annotation_mask.max() + 1):
            instance_mask = (current_annotation_mask == mask_id).astype(np.uint8) * 255
            current_masks.append(instance_mask)

        # 3. compute DINOv2
        aff = self.compute_cost_volume(current_obs_image, ref_image, h, w)

        # 3. compute iou to find the mask
        max_mask_id = ref_annotation_mask.max()
        corresponding_mask_ids = {}
        new_current_annotation_mask = np.zeros_like(ref_annotation_mask)

        current_binary_object_mask = (current_annotation_mask > 0)
        current_patchified_mask = self.get_patchified_mask(current_binary_object_mask, patch_size)


        hungarian_matrix = np.zeros((max_mask_id, current_annotation_mask.max()))
        for mask_id in range(1, max_mask_id + 1):
            # 3.1 Get the binary mask of the current mask_id from reference annotation
            instance_mask = (ref_annotation_mask == mask_id).astype(np.float32)
            resized_instance_mask = resize_image_to_same_shape(instance_mask, ref_image)

            # 3.2 Patchify the mask
            ref_patchified_mask = self.get_patchified_mask(resized_instance_mask, patch_size)

            # This is to decide the patches that are on the line of mask boundaries. We assume 0.5 occupancy is considered as in the mask.
            new_mask = ref_patchified_mask > threshold
            mask_indices = np.where(new_mask == 1)

            overlapped_indices = []

            # mask_ids[0] is an array of x coordinate, mask_ids[1] is an array of y coordinate
            for (i, j) in zip(mask_indices[0], mask_indices[1]):
                select_aff = aff[i, j]
                select_aff_h, select_aff_w = select_aff.shape
                image_flat = select_aff.reshape(select_aff_h * select_aff_w, 1) / temperature

                # filter_mask will set the affinity of the patches that are not within object masks to 0.
                filter_mask = torch.tensor(current_patchified_mask.reshape(select_aff_h * select_aff_w, 1) > threshold).float()
                image_flat = image_flat * filter_mask
                softmax = torch.exp(image_flat) / torch.sum(torch.exp(image_flat), axis=0)
                select_aff = softmax.reshape(select_aff_h, select_aff_w)

                topk_threshold = softmax[softmax.squeeze().argsort(descending=True)[topk]].detach().cpu().numpy()
                select_aff = rescale_feature_map(select_aff.unsqueeze(0).unsqueeze(0), current_obs_image.shape[0], current_obs_image.shape[1]).squeeze()

                binary_mask = select_aff > topk_threshold

                iou_score = compute_iou(binary_mask, np.stack(current_masks, axis=0))

                overlapped_indices.append(iou_score.argsort()[::-1][0])
            corresponding_mask_ids[mask_id] = find_most_repeated_number(overlapped_indices)
            # print(mask_id, ": ", overlapped_indices)
            # print(mask_id, ": ", corresponding_mask_ids[mask_id])
            # print(overlapped_indices)
            # print("----------------")
            for matched_id in overlapped_indices:
                hungarian_matrix[mask_id-1, matched_id-1] += 1
            hungarian_matrix[mask_id-1, :] /= np.sum(hungarian_matrix[mask_id-1, :])

            new_mask_resized_as_ref_seg = resize_image_to_same_shape(current_masks[mask_id], ref_image)
            new_current_annotation_mask[np.where(new_mask_resized_as_ref_seg == 255)] = mask_id
        # new_current_annotation_mask = resize_image_to_same_shape(new_current_annotation_mask, current_obs_image_input)
        return new_current_annotation_mask, hungarian_matrix


    def spatial_correspondence(self,
                                src_image, 
                                tgt_image,
                                query_points,
                                src_annotation=None,
                                tgt_annotation=None,
                                h=32,
                                w=32,
                                temperature=100,
                                sizes=[448, 336, 224],
                                weights=[1., 0.2, 0.2]):
        current_obs_image = resize_image_to_same_shape(tgt_image, src_image)
        img_list = []
        feature_list = []
        for img in [src_image, current_obs_image]:
            img_list.append(img)
            feature_list.append(self.dinov2.process_image(img, sizes=sizes, weights=weights))
        aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w), temperature=temperature)

        aff = rescale_feature_map(aff, tgt_image.shape[0], tgt_image.shape[1])
        corresponding_points = []
        for point in query_points:
            patch_x, patch_y = math.floor(point[0] / src_image.shape[1] * aff.shape[0]), math.floor(point[1] / src_image.shape[0] * aff.shape[1])

            selected_aff = aff[patch_y, patch_x]

            # filter attention that are outside of object boundary
            if tgt_annotation is not None:
                selected_aff = selected_aff * (tgt_annotation > 0)

            argmax_y, argmax_x = np.unravel_index(selected_aff.argmax(), selected_aff.shape)
            corresponding_points.append([argmax_x, argmax_y])
        return aff, feature_list, corresponding_points


# class CorrespondenceOperator():
#     def __init__(self):
#         self.dinov2 = DinoV2ImageProcessor()

#     def spatial_correspondence(self,
#                                 src_image, 
#                                 tgt_image,
#                                 query_points,
#                                 src_annotation=None,
#                                 tgt_annotation=None,
#                                 h=32,
#                                 w=32,
#                                 temperature=100,
#                                 sizes=[448, 224],
#                                 weights=[1., 0.3]):
#         current_obs_image = resize_image_to_same_shape(tgt_image, src_image)
#         img_list = []
#         feature_list = []
#         for img in [src_image, current_obs_image]:
#             img_list.append(img)
#             feature_list.append(self.dinov2.process_image(img, sizes=sizes, weights=weights))
#         aff = compute_affinity((feature_list[0], h, w), (feature_list[1], h, w), temperature=temperature)

#         aff = rescale_feature_map(aff, tgt_image.shape[0], tgt_image.shape[1])
#         corresponding_points = []
#         for point in query_points:
#             patch_x, patch_y = math.floor(point[0] / src_image.shape[1] * aff.shape[0]), math.floor(point[1] / src_image.shape[0] * aff.shape[1])

#             selected_aff = aff[patch_y, patch_x]

#             # filter attention that are outside of object boundary
#             if tgt_annotation is not None:
#                 selected_aff = selected_aff * (tgt_annotation > 0)

#             argmax_y, argmax_x = np.unravel_index(selected_aff.argmax(), selected_aff.shape)
#             corresponding_points.append((argmax_x, argmax_y))
#         return aff, feature_list, corresponding_points
