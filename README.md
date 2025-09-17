https://github.com/Hiro1301/anomaly-prediction-research/new/main
# Anomaly Prediction Research

This repository contains code and configuration for a graduation research project on premonitory anomaly detection in surveillance camera footage. The objective is to detect signs of abnormal events several seconds to tens of seconds before they occur and evaluate both detection performance (AUC/AP) and timeliness (earliness/TTD).

## Directory Structure

```
project/
  data/                 # clipped video data
  ann/                  # ground truth and premonitory labels
  features/
    clip/               # CLIP embeddings (.npy)
    yolo/               # YOLO/DETR detection results (JSON)
    stats/              # brightness/crowd/motion features (.parquet)
  models/
    context/            # environment models
    skeleton/           # pose and interaction models
    fusion/             # fusion models
  configs/              # YAML configuration files
  scripts/              # preprocessing, training and evaluation scripts
  notebooks/            # Jupyter notebooks for analysis and visualization
  apps/                 # real-time inference applications
```

## Getting Started

1. **Install dependencies** using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

2. **Build clips** from raw video with `build_clips.py` and the dataset YAML configuration.

3. **Generate premonitory labels** with `make_premonitory_labels.py`. For example:
    ```bash
    python scripts/make_premonitory_labels.py --cfg configs/dataset_ucsd.yaml --deltas 5 10 30 60
    ```

4. **Extract features** using CLIP, YOLO and compute environment statistics:
    ```bash
    python scripts/extract_clip_feats.py --cfg configs/dataset_ucsd.yaml --model clip-vit-b32
    python scripts/run_yolo.py --cfg configs/dataset_ucsd.yaml --classes person bicycle car bag tool
    python scripts/make_env_features.py --cfg configs/dataset_ucsd.yaml
    ```

5. **Train baseline** environment model and evaluate:
    ```bash
    python scripts/train_env_baseline.py --cfg configs/train_env_ucsd.yaml --seed 3407 --repeat 5
    python scripts/eval.py --cfg configs/eval_ucsd.yaml --metrics auc ap earliness@0.8 ttd far
    ```

See individual scripts for more details.

## License

This project is intended for research purposes only.
