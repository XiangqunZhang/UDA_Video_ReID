## From Synthetic to Real: Unveiling the Power of Synthetic Data for Video Person Re-ID

This document provides instructions on how to use the codebase for video-based person re-identification.

### Requirements: Python=3.7 and Pytorch=1.8.1

### Data preparation
Download the dataset from this link [**SVReID**](https://drive.google.com/file/d/12LCms0nW0ipUmDEk2Xg8t8GmiGFLysew/view?usp=sharing).

### Getting Started

To train the model for unsupervised domain adaptation:

```shell
# Train
python main_SSA.py \
    --dataset ${svreid, svreid_cc, svreid_plus, mars, ...} \
    --root ${path of source dataset} \
    --td ${ilidsvid, ccvid, ...} \
    --tdroot ${path of target dataset} \
    --save_dir ${path for saving logs} \
    --gpu_devices 0,1,2,3
```

To test the trained model using all frames:

```shell
# Test with all frames
python main.py \
    --dataset ${svreid, svreid_cc, svreid_plus, mars, ...} \
    --root ${path of source dataset} \
    --td ${ilidsvid, ccvid, ...} \
    --tdroot ${path of target dataset} \
    --save_dir ${path for saving logs} \
    --evaluate \
    --resume ${path of trained model}
    --gpu_devices 0,1,2,3 \
```

---

Please make sure to replace `${...}` placeholders with the actual paths and configurations relevant to your setup.

### Acknowledgments

This code is based on the implementations of [**SINet**](https://github.com/baist/SINet).