

## 1. Installation

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


## 2. Data Preparation

### Record the video and export the file
1. Record the video using the `Record3D` app.


2. Export the file from the app. The exported file will be in `.r3d` format.

3. Transfer the r3d file to your computer, and do the following:
```
python scripts/export_from_r3d.py --r3d-path R3D_PATH --depth --video
```

If everything is correct, you would be able to see a `hdf5` file located under a repo with the same name as the r3d file, let's refer to it as `DATASET_FOLDER`.


### Annotate the video with object-centric information

1. Grounded-SAM

Annotate the first frame image from the dataset located at `DATASET_FOLDER` using the following command:
```
python scripts/01_gam_annotation.py

```

Or use the jupyter notebook if you want to avoid repetitive loading Grounded-SAM model, which is quite time-consuming.


2. Preprocess VOS, TAP, and the temporal segmentation
```
    python scripts/02_preprocess_video.py --annotation-path ANNOTATION_FOLDER  --tap-pen TAP_PEN
```

`--tap-pen` is an argument to specify the penalty used for TAP annotation. It's a hyperparameter required for changepoint detection.


### Annotate the video with human hand information

```
python scripts/03_hand_analysis.py --annotation-path ANNOTATION_FOLDER
```

We can visualize the hand result by running the following script:
```
python scripts/hand_visualization.py --annotation-path ANNOTATION_FOLDER
```

### Visualization
To make sure the data preparation makes sense, we provide information on double checking if the data is correct.

The annotation folder should look like this:
```
annotaitons/
    human_demo/
        DATASET_Name/
            frame.jpg             # first frame of the video
            frame_annotation.png  # first frame segmentaiton annotation from Grounded-SAM
            masks.npz             # XMem annotation result

            # TAP annotation result
            points.pt          # sampled keypoints
            tracked_points.pt  # tracked tracks and visibility information

            tap_temporal_segments.pt         # changepoint detection result
            hammer_output/hand_keypoints.pt  # HaMeR result

            # Visualization purpose
            xmem_annotation_video.mp4 # XMem annotation video
            cotracker/video.mp4       # TAP annotation video
            tap_segmentation.mp4      # changepoint detection visualization
            hamer_recon.mp4           # hamer visualization
```

Then use the following jupyter notebook to visualize the result `visualize_data.ipynb`. 


### Visualize everything together
Play with `visualization_example.ipynb` in the parent folder.

## 3. Real Robot Rollout
TODO

### Camera Calibration


### Run Camera

```
python real_robot_scripts/cam_streaming.py --camera-ref rs_0 --use-rgb --use-depth --eval --use-rec --visualization
```