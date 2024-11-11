# RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving

This is the official PyTorch implementation of the paper **RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving**, by Gopi Krishna Erabati and Helder Araujo.

G. K. Erabati and H. Araujo, "RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving," in _Computer Vision and Image Understanding (CVIU)_, 2024.

**Contents**
1. [Overview](https://github.com/gopi-erabati/RetSeg3D#overview)
2. [Results](https://github.com/gopi-erabati/RetSeg3D#results)
3. [Requirements, Installation and Usage](https://github.com/gopi-erabati/RetSeg3D#requirements-installation-and-usage)
    1. [Prerequistes](https://github.com/gopi-erabati/RetSeg3D#prerequisites)
    2. [Installation](https://github.com/gopi-erabati/RetSeg3D#installation)
    3. [Training](https://github.com/gopi-erabati/RetSeg3D#training)
    4. [Testing](https://github.com/gopi-erabati/RetSeg3D#testing)
4. [Acknowledgements](https://github.com/gopi-erabati/RetSeg3D#acknowlegements)
5. [Reference](https://github.com/gopi-erabati/RetSeg3D#reference)

## Overview
LiDAR semantic segmentation is one of the crucial tasks for scene understanding in autonomous driving. Recent trends suggest that voxel- or fusion-based methods obtain improved performance. However, the fusion-based methods are computationally expensive. On the other hand, the voxel-based methods uniformly employ local operators (e.g., 3D SparseConv) without considering the varying-density property of LiDAR point clouds, which result in inferior performance, specifically on far away sparse points due to limited receptive field. To tackle this issue, we propose novel retention block to capture long range dependencies and maintain the receptive field of far away sparse points and design **RetSeg3D**, a retention-based 3D semantic segmentation model for autonomous driving. Instead of vanilla attention mechanism to model long range dependencies, inspired by RetNet, we design cubic window multi-scale retentive self-attention (CW-MSRetSA) module with bidirectional and 3D explicit decay mechanism to introduce 3D spatial distance related prior information into the model to improve not only the receptive field but also the model capacity. Our novel retention block maintains the receptive field which significantly improve the performance of far away sparse points. We conduct extensive experiments and analysis on three large-scale datasets: SemanticKITTI, nuScenes and Waymo. Our method not only outperform existing methods on far away sparse points but also on close and medium distance points and efficiently runs in real time at 52.1 FPS.

![RetSeg3D_arch](https://github.com/gopi-erabati/RetSeg3D/assets/22390149/f7afe137-316a-4337-bcba-45e07a606ada)

## Results

### Predictions on Waymo dataset
![1724336343405-ezgif com-optimize](https://github.com/user-attachments/assets/29e52396-7573-4908-9c75-67a1cb5010c3)

### Predictions on SemanticKITTI, Waymo and nuScenes datasets
![RetSeg3D_Retention-based3DSemanticSegmentationforAutonomousDriving-ezgif com-video-to-gif-converter](https://github.com/gopi-erabati/RetSeg3D/assets/22390149/254570b0-0cfb-49cc-961a-6be1bfbab68a)

### Quantiative Results (mIoU)

| RetSeg3D | SemanticKITTI | nuScenes | Waymo |
| :---: | :---: | :---: | :---: |
| mIoU | 70.3 | 76.9 | 70.1 |
| Config | retseg3d_semantickitti.py | retseg3d_nus.py | retseg3d_waymo.py |
| Model | [weights](https://drive.google.com/file/d/1fK4c0lGLiDX5jjpmEEwpEL_fyES3CdMz/view?usp=sharing) | [weights](https://drive.google.com/file/d/1SehjzIpXr-nTbH6EpEUTvzwMYJ_Hl4D3/view?usp=sharing) | |

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

## Requirements, Installation and Usage

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==2.0.1
- [mmcv](https://github.com/open-mmlab/mmcv)==2.1.0
- [mmengine](https://github.com/open-mmlab/mmengine)==0.10.1
- [mmdet](https://github.com/open-mmlab/mmdetection)==3.2.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.3.0

### Installation
```
mkvirtualenv retseg3d

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -U openmim
mim install mmcv==2.1.0
mim install mmengine==0.10.1

pip install -r requirements.txt
```

### Data
Follow [MMDetection3D-1.3.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0) to prepare the [SemanticKITTI](https://mmdetection3d.readthedocs.io/en/v1.3.0/advanced_guides/datasets/semantickitti.html) and [nuScenes](https://mmdetection3d.readthedocs.io/en/v1.3.0/advanced_guides/datasets/nuscenes.html) datasets. Follow [Pointcept](https://github.com/Pointcept/Pointcept) for Waymo data prepreocessing and then run `python tools/create_waymo_semantic_info.py /path/to/waymo/preprocess/dir` to generate the `.pkl` files required for the config. 

**Warning:** Please strictly follow [MMDetection3D-1.3.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0) code to prepare the data because other versions of MMDetection3D have different coordinate refactoring.

### Clone the repository
```
git clone https://github.com/gopi-erabati/RetSeg3D.git
cd RetSeg3D
```

### Training

#### SemanticKITTI dataset
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/retseg3d_semantickitti.py --work-dir {WORK_DIR}`.
- Multi GPU training
  `tools/dist_train.sh configs/retseg3d_semantickitti.py {GPU_NUM} --work-dir {WORK_DIR}`.
  
#### nuScenes dataset
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/retseg3d_nus.py --work-dir {WORK_DIR}`.
- Multi GPU training
  `tools/dist_train.sh configs/retseg3d_nus.py {GPU_NUM} --work-dir {WORK_DIR}`.

#### Waymo dataset 
- Single GPU training
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/train.py configs/retseg3d_waymo.py --work-dir {WORK_DIR}`
- Multi GPU training
  `tools/dist_train.sh configs/retseg3d_waymo.py {GPU_NUM} --work-dir {WORK_DIR}`

### Testing

#### SemanticKITTI dataset
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/retseg3d_semantickitti.py /path/to/ckpt --work-dir {WORK_DIR}`
- Multi GPU testing `./tools/dist_test.sh configs/retseg3d_semantickitti.py /path/to/ckpt {GPU_NUM} --work-dir {WORK_DIR}`.

#### nuScenes dataset
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/retseg3d_nus.py /path/to/ckpt --work-dir {WORK_DIR}`.
- Multi GPU testing `./tools/dist_test.sh configs/retseg3d_nus.py /path/to/ckpt {GPU_NUM} --work-dir {WORK_DIR}`.

#### Waymo dataset 
- Single GPU testing
    1. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
    2. `python tools/test.py configs/retseg3d_waymo.py /path/to/ckpt --work-dir {WORK_DIR}`.
- Multi GPU testing `./tools/dist_test.sh configs/retseg3d_waymo.py /path/to/ckpt {GPU_NUM} --work-dir {WORK_DIR}`.

## Acknowlegements
We sincerely thank the contributors for their open-source code: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [Pointcept](https://github.com/Pointcept/Pointcept).

## Reference
```
@article{retseg3d,
title = {RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving},
journal = {Computer Vision and Image Understanding},
year = {2024},
author = {Gopi Krishna Erabati and Helder Araujo},
}
```
