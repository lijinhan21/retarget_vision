import cv2
import sys
import json

import numpy as np
from argparse import ArgumentParser

import shutil
import torch

import matplotlib.pyplot as plt
from easydict import EasyDict

from orion.utils.misc_utils import (
    load_first_frame_from_hdf5_dataset, 
    export_video_from_hdf5_dataset,
    load_first_frame_from_human_hdf5_dataset, 
    export_video_from_human_hdf5_dataset,
    overlay_xmem_mask_on_image
    )
from orion.algos.grounded_sam_wrapper import GroundedSamWrapper


torch.set_grad_enabled(False)

wrapper = GroundedSamWrapper()
