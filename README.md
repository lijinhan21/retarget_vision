

## 1. Installation

### Create a python environment
```
conda create --name orion python=3.9
```

Then first install the latest torch in this virtual environment before you proceed. This is to make sure `torch` can correctly installed with cuda enabled. Note that you should create a separate virtualenv in order to run HaMeR.

### Install DINOv2, SAM, XMem, Cotracker, Cutie, Hand Object Detector
1. Run `./setup_vision_models.sh` to install the following packages:
    - DINOv2
    - SAM
    - XMem
    - Cotracker
    - Cutie
    - Hand Object Detector

<!-- ### Install HaMeR
Follow the instruction at [HaMeR](https://github.com/geopavlakos/hamer), and install it under `third_party`.

Then, in order to run the model in our repo, we need to modify the path of some folders. The following step is recommended under the project root directory:
```
ln -n -s third_party/hamer/_DATA ./
```

Also, change line 14 of `hamer/vitpose_model.py` to the following:
```
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
```

Note that it's recommended to use a separate conda environment to run the HaMeR code. There are some package version conflicts that make it difficult to run HaMeR in the same environment as the rest of the code. Luckily, this part is only for offline processing, so it's won't affect workflow at test time. 
 -->



### Install Grounded-SAM

Git clone [Grounded-SAM](https://github.com/IDEA-Research/GroundingDINO) under `third_party`, and follow its instructions.

If you want to use the grounded-sam with GPU, things might not be installed correctly following the website. One possible solution is to run the following commands separately:
```
python setup.py build
python setup.py install
```

### Install Lietorch

You can install the following anywhere, as long as you install it into the `orion` virtual environment.
```
git clone git@github.com:princeton-vl/lietorch.git

```

### Directory structure of `third_party`

The resulting directory structure should look like this:

```
_DATA # for hamer
third_party/
    co-tracker/
    cutie/
    dinov2/
    segment-anything/
    GroundingDINO/
    hand_object_detector/
    sam_checkpoints/
    segment-anything/
    XMem/
    xmem_checkpoints/
```

## 2. Data Preparation (scripts_new)

### Generate from RGB videos

First, put your video inside `datasets/iphones`, then generate hdf5 file
```
python scripts_new/create_hdf5_from_rgb.py --filepath PATH_TO_VIDEO_RELATIVE_TO_datasets/iphones
```

Then run pipeline to do temporal segmentation.
```
python scripts_new/pipeline.py --human_demo PATH_TO_HDF5_RELATIVE_TO_datasets/iphones --no-depth --no-smplh
```

All generated results are in `annotation\human_demo` folder.

### Generate from RGBD videos

Assume that you already convert rgbd video into a hdf5 file. Put the hdf5 file under `datasets/iphones`.

If you have smplh file, run 
```
MUJOCO_GL="glx" python scripts_new/pipeline.py --human_demo PATH_TO_HDF5_RELATIVE_TO_datasets/iphones --smplh-path PATH_TO_SMPLH
``` 
If you do not have smplh file, run
```
MUJOCO_GL="glx" python scripts_new/pipeline.py --human_demo PATH_TO_HDF5_RELATIVE_TO_datasets/iphones --no-smplh
```

### Details

1. Generate text descriptions of objects of interest in the video by VLM.
2. Run GAM segmentation on the first frame.
3. Use cutie to do VOS.
4. Sample keypoints, and use cotracker to track them.
5. Do temporal segmentation.
6. Prepare to generate hoig:
    - Generate human smplh trajectory from video
    - Determine whether need waypoints for each step by VLM. (Output 'Target Only' or 'Whole Trajectory')
    - TODO: calculate ik weights by VLM, and do hand object contact detection by hand_object_detector.
7. Generate hoig from video, retarget and apply ik (with grasp primitives), and visualize the plan in `plan_vis.mp4` in annotation folder.

## 2. Data Preparation (scripts)

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

Make sure you run `03_hand_analysis.py` in the virtualenvironment for HaMeR.
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