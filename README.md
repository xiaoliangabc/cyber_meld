# Map-Enhanced Ego-Lane Detection in the Missing Feature Scenarios

This repository provides the official implementation for "[Map-Enhanced Ego-Lane Detection in the Missing Feature Scenarios](https://ieeexplore.ieee.org/document/9110871?source=authoralert)".

## Abstract

As one of the most important tasks in autonomous driving systems, ego-lane detection has been extensively studied and has achieved impressive results in many scenarios. However, ego-lane detection in the missing feature scenarios is still an unsolved problem. To address this problem, previous methods have been devoted to proposing more complicated feature extraction algorithms, but they are very time-consuming and cannot deal with extreme scenarios. Different from others, this paper exploits prior knowledge contained in digital maps, which has a strong capability to enhance the performance of detection algorithms. Specifically, we employ the road shape extracted from OpenStreetMap as lane model, which is highly consistent with the real lane shape and irrelevant to lane features. In this way, only a few lane features are needed to eliminate the position error between the road shape and the real lane, and a search-based optimization algorithm is proposed. Experiments show that the proposed method can be applied to various scenarios and can run in real-time at a frequency of 20 Hz. At the same time, we evaluated the proposed method on the public KITTI Lane dataset where it achieves state-of-the-art performance.

## Dependencies

This code is implemented in C++ with the following packages:
1. pcl
2. OpenCV
3. Eigen3
4. OpenMP
5. [Fade2D](https://www.geom.at/fade25d/html/)

## Setup

### Download Code

```bash
git clone git@github.com:xiaoliangabc/cyber_meld.git
```

### Download Data

Download data from [google driver](https://drive.google.com/file/d/1TfXMpyTRzB1Vgkiy78c5oaF_JZ5uH57P/view?usp=sharing) and unzip it into `data` folder

```bash
cd cyber_meld
unzip kitti_lane_data.zip -d ./data/
rm kitti_lane_data.zip
```

## Usage

### Build

```bash
mkdir build
cd build
cmake ..
make
```

### Run

#### Traning Dataset

```
./ego_lane_detection training
```

#### Testing Dataset

```
./ego_lane_detection testing
```

### Evaluate

```bash
cd evaluation
python evaluateRoad.py ../data/training/result/ ../data/training/gt_bev_image/
```

The following information will be output to the console

```
MaxF: 92.94 
AvgPrec: 85.44 
PRE_wp: 92.65 
REC_wp: 93.24 
FPR_wp: 1.14 
FNR_wp: 6.76
```

## Other

If you have GPU support, we highly recommend using PLARD's road detection results as ROI, which produce higher accuracy. For more information, please refer to the [plard](https://github.com/xiaoliangabc/cyber_meld/tree/plard) branch.

## Citation

If you find this work useful, please cite:

```
@ARTICLE{9110871,
  author={X. {Wang} and Y. {Qian} and C. {Wang} and M. {Yang}},
  journal={IEEE Access}, 
  title={Map-Enhanced Ego-Lane Detection in the Missing Feature Scenarios}, 
  year={2020},
  volume={8},
  pages={107958-107968}
}
```

## Acknowledgement

The PLARD result is obtained from [PLARD](https://github.com/zhechen/PLARD).
