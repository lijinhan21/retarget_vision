cd third_party

git clone git@github.com:hkchengrex/XMem.git

cd XMem

# inside third_party/XMem/
bash scripts/download_models_demo.sh

# back to third_party/
cd ..

mkdir xmem_checkpoints

mv XMem/saves/*pth xmem_checkpoints/

# Cutie, as a replacement of XMem
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
pip install -e .

python scripts/download_models.py
cd ..


# cotracker
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
cd ..


# dinov2
git clone git@github.com:facebookresearch/dinov2.git
cd dinov2
git checkout fc49f49d734c767272a4ea0e18ff2ab8e60fc92d
pip install -r requirements.txt
pip install -e .

cd ..

# SAM
git clone https://github.com/facebookresearch/segment-anything
cd segment-anything
git checkout 6fdee8f2727f4506cfbbe553e23b895e27956588
pip install -e .

cd ..
mkdir sam_checkpoints
cd sam_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth