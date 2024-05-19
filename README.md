# RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving

This is the official PyTorch implementation of the paper **RetSeg3D: Retention-based 3D Semantic Segmentation for Autonomous Driving**, by Gopi Krishna Erabati and Helder Araujo.

**Contents**
1. Overview

## Overview
LiDAR semantic segmentation is one of the crucial tasks for scene understanding in autonomous driving. Recent trends suggest that voxel- or fusion-based methods obtain improved performance. However, the fusion-based methods are computationally expensive. On the other hand, the voxel-based methods uniformly employ local operators (e.g., 3D SparseConv) without considering the varying-density property of LiDAR point clouds, which result in inferior performance, specifically on far away sparse points due to limited receptive field. To tackle this issue, we propose novel retention block to capture long range dependencies and maintain the receptive field of far away sparse points and design **RetSeg3D**, a retention-based 3D semantic segmentation model for autonomous driving. Instead of vanilla attention mechanism to model long range dependencies, inspired by RetNet, we design cubic window multi-scale retentive self-attention (CW-MSRetSA) module with bidirectional and 3D explicit decay mechanism to introduce 3D spatial distance related prior information into the model to improve not only the receptive field but also the model capacity. Our novel retention block maintains the receptive field which significantly improve the performance of far away sparse points. We conduct extensive experiments and analysis on three large-scale datasets: SemanticKITTI, nuScenes and Waymo. Our method not only outperform existing methods on far away sparse points but also on close and medium distance points and efficiently runs in real time at 52.1 FPS.

![RetSeg3D_arch](https://github.com/gopi-erabati/RetSeg3D/assets/22390149/f7afe137-316a-4337-bcba-45e07a606ada)

### Predictions on SemanticKITTI, Waymo and nuScenes datasets
![RetSeg3D_Retention-based3DSemanticSegmentationforAutonomousDriving-ezgif com-video-to-gif-converter](https://github.com/gopi-erabati/RetSeg3D/assets/22390149/254570b0-0cfb-49cc-961a-6be1bfbab68a)

### Quantiative Results (mIoU)

| RetSeg3D | SemanticKITTI | nuScenes | Waymo |
| :---: | :---: | :---: | :---: |
| mIoU | 70.3 | 76.9 | 70.1 |
| Config | retseg3d_semantickitti.py | retseg3d_nus.py | retseg3d_waymo.py |
| Model | weights | weights | |

We can not distribute the model weights on Waymo dataset due to the [Waymo license terms](https://waymo.com/open/terms).

### Prerequisites

The code is tested on the following configuration:
- Ubuntu 20.04.6 LTS
- CUDA==11.7
- Python==3.8.10
- PyTorch==2.0.1
- [mmcv](https://github.com/open-mmlab/mmcv)==2.1.0
- [mmdet](https://github.com/open-mmlab/mmdetection)==3.2.0
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==1.3.0



