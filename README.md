## From Synthetic to Real: Unveiling the Power of Synthetic Data for Video Person Re-ID

This document provides instructions on how to use the codebase for video-based person re-identification.

### Getting Started

#### UDA Video ReID (Unsupervised Domain Adaptation)

To train the model for unsupervised domain adaptation:

```shell
# Train
python main_SSA.py \
    --root ${path of source dataset} \
    --td ilidsvid \
    --save_dir ${path for saving logs} \
    --gpu_devices 0,1,2,3
```

#### Cross-Domain Video ReID Baseline

To train the baseline model for cross-domain video re-identification:

```shell
# Train
python main.py \
    --dataset ${mars, lsvid, ...} \
    --root ${path of dataset} \
    --gpu_devices 0,1,2,3 \
    --save_dir ${path for saving models and logs}
```

To test the trained model using all frames:

```shell
# Test with all frames
python main.py \
    --dataset mars \
    --root ${path of dataset} \
    --gpu_devices 0,1,2,3 \
    --save_dir ${path for saving logs} \
    --evaluate \
    --all_frames \
    --resume ${path of pretrained model}
```

---

Please make sure to replace `${...}` placeholders with the actual paths and configurations relevant to your setup.