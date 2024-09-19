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

To test the trained model:

```shell
# Test
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

### Model Weights
The model weights of major cross-domain experiments are provided here. The results in the paper can be obtained through the test script. The table lists the major cross-domain directions and rank-1 and mAP values.


| Types                           |    Paper    | Reproduce |                                           Download                                           |
|---------------------------------|:-----------:|:---------:|:--------------------------------------------------------------------------------------------:| 
| SVReID→iLIDS-VID          |  42.0/53.3  | 43.3/54.7 | [model](https://drive.google.com/file/d/1p7lwBdojaigv9_4d7iQC2nRvRvp_CNcp/view?usp=sharing)  | 
| SVReID→PRID          |  71.9/79.1  |     -     | [model](https://drive.google.com/file/d/13zSFhHkaDnxsbF0MBuCl6S0d4oMA7_No/view?usp=sharing)  |
| SVReID→Mars |  48.1/30.8  |     -     | [model](https://drive.google.com/file/d/1pG8z7D4FYN8iyE54nruWWOOE4w1fUNjU/view?usp=sharing)  | 
| SVReID→LS-VID  |  20.1/8.6   |     -     | [model](https://drive.google.com/file/d/1t6Un-xGvSZhv1lwbNHDKPNdALzkjfLNm/view?usp=sharing)  | 
| SVReID (w cc)→CCVID |  78.3/72.4  |     -     | [model](https://drive.google.com/file/d/19gfsvQAFVxszpq08R4F1bdr5HiEwhLKM/view?usp=sharing)  | 
