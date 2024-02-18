

## Installation

### Install DINOv2, SAM, XMem, Cotracker
1. Run `./setup_vision_models.sh` to install the following packages:
    - DINOv2
    - SAM
    - XMem
    - Cotracker

### Install HaMeR
Follow the instruction at [HaMeR](https://github.com/geopavlakos/hamer), and install it under `third_party`.

Then, in order to run the model in our repo, we need to modify the path of some folders. The following step is recommended under the project root directory:
```
ln -n -s third_party/hamer/_DATA ./
```

Note that it's recommended to use a separate conda environment to run the HaMeR code. There are some package version conflicts that make it difficult to run HaMeR in the same environment as the rest of the code. Luckily, this part is only for offline processing, so it's won't affect workflow at test time. 


### Install Grounded-SAM

Git clone [Grounded-SAM](https://github.com/IDEA-Research/GroundingDINO) under `third_party`, and follow its instructions.

If you want to use the grounded-sam with GPU, thigns might not be installed correctly following the website. One possible solution is to run the following commands separately:
```
python setup.py build
python setup.py install
```

### Directory structure of `third_party`

The resulting directory structure should look like this:

```
_DATA # for hamer
third_party/
    co-tracker/
    dinov2/
    segment-anything/
    hamer/
    Grounded-Segment-Anything/
    sam_checkpoints/
    XMem/
    xmem_checkpoints/
```


## Data Preparation

### Annotate the video with object-centric information




### Annotate the video with human hand information

```
python scripts/hand_analysis.py --annotation-path ANNOTATION_FOLDER
```
```
python scripts/hand_visualization.py --annotation-path ANNOTATION_FOLDER
```



## Naming
| Name | Description |
| --- | --- |
| `seq` | Refer to the sequence |
| `stereo_seq` | the sequence of stereo images, [rightview_sequence, leftview_sequence] |
| `rightview_seq` | the sequence of right view images, (T, H, W, C) |
| `leftview_seq` | the sequence of left view images, (T, H, W, C) |
| `video_seq` | the sequence of video frames, (T, H, W, C) |
| `feature_seq` | the sequence of feature maps, (T, D), D could be multi-dimensional, depends on how the feature is encoded|
| `rightview_feature_seq` | the sequence of right view feature maps, (T, D), D could be multi-dimensional, depends on how the feature is encoded|
| `temporal_segments` | temporal segmetns in BUDS, (N, 2), N is the number of segments, 2 is the start and end frame index |



First todo:
1. Get a sequence of stero images
2. Run point tracker on objects
3. Visualize the vectors in 3D space
4. Implement functionalities to easily retrieve data from the sequence given the `temporal_segments``

Second todo:
1. Set up ZED2
2. Get a sequence of stero images from ZED2
3. Repeat the other process from first todos

Third todo:
~~1. Run BUDS on the image sequence~~

## Experiment Runs:
1. Launch `prototype_experiment_management.ipynb`
    a. select a task
    b. select a model
    c. create experiment

2. 



## Example usage

### BUDS Segmentation
```
# initialize buds_segmentation_cfg from hydra config
buds_segmentation_cfg = ...

# initialize the object for BUDS segmentation
buds_segmentation = BUDSSegmentation(buds_segmentation_cfg)

# reset the internal states in BUDS segmentation
buds_segmentation.reset()

# if video_seq is None, it assumes some features are saved temporarily. 
segmentation_result = buds_segmentation(
    video_seq=video_seq,
    save_feature=True,
    cfg=buds_segmentation_cfg,
)

# segmentation_result is a dictionary with the following keys
# segmentation_result = {
#     'temporal_segments': temporal_segments,
#     'feature_seq': feature_seq,
#     'node_list': node_list,
#     }
```


### First Frame Annotation on Objects
1. Scribble2Mask Annotation
If it's a single image:
```
python scripts/interactive_demo_example.py --image IMAGE_PATH --num_objects NUM_OBJECTS
```

If it's a video:
```
python scripts/interactive_demo_example.py --video VIDEO_PATH --num_objects NUM_OBJECTS
```


1. Generate keypoints from annotation
```
python scripts/generate_keypoints_from_annotation.py --annotation-folder ANNOTATION_FOLDER
```

1. Track points using cotracker
```
python scripts/cotracker_annotation.py --annotation-path ANNOTATION_FOLDER
```

### Hamer 
```
python scripts/hand_analysis.py --annotation-path ANNOTATION_FOLDER
```

```
python scripts/hand_visualization.py --annotation-path ANNOTATION_FOLDER
```


### Tracking Objects
1. (TODO) VOS Annotation



2. (TODO) Point Tracking



### Create a dataset from r3d file
```
python scripts/export_from_r3d.py --r3d-path R3D_PATH --depth --video
```

If using front camera for recording, add `--front-camera` flag. Otherwise, it will use the back camera by default, which has different resolution.

Note: The camera intrinsics in metadata is for color images. Therefore, in order to get successful point cloud reconstruction, it's suggested to resize depth image (originally 256 x 192) to color image size 1920x1440.
