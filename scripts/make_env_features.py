#!/usr/bin/env python3
"""
Compute environment features for anomaly premonition detection.
This script extracts brightness, crowd density, and motion features
from each clip using available CLIP embeddings, YOLO detection results,
and raw video frames. The resulting features are saved in Parquet format.

The implementation provided here is a baseline skeleton. You can extend
or replace the feature computation functions with more sophisticated
methods as needed. The goal is to provide a reproducible and modular
pipeline for environment context extraction.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2  # OpenCV for video processing
import numpy as np
import pandas as pd
import yaml


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def compute_brightness(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute brightness statistics from a list of frames.

    Brightness is approximated by the mean pixel intensity of each frame.
    Returns the mean, standard deviation, min, and max across all frames.
    """
    if not frames:
        return {"brightness_mean": 0.0, "brightness_std": 0.0, "brightness_min": 0.0, "brightness_max": 0.0}
    intensities = [float(frame.mean()) for frame in frames]
    return {
        "brightness_mean": float(np.mean(intensities)),
        "brightness_std": float(np.std(intensities)),
        "brightness_min": float(np.min(intensities)),
        "brightness_max": float(np.max(intensities)),
    }


def compute_motion(frames: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute motion statistics based on frame-to-frame differences.

    Motion energy is measured as the mean absolute difference between
    consecutive frames. Returns mean, std, and max motion values.
    """
    if len(frames) < 2:
        return {"motion_mean": 0.0, "motion_std": 0.0, "motion_max": 0.0}
    motions = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        motions.append(float(diff.mean()))
    return {
        "motion_mean": float(np.mean(motions)),
        "motion_std": float(np.std(motions)),
        "motion_max": float(np.max(motions)),
    }


def compute_crowd(yolo_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute crowd density statistics from YOLO detection results.

    Expects a list of detection dictionaries, one per frame, where each
    dictionary contains a list of detected objects with class labels.
    Counts the number of "person" detections per frame and returns
    statistics over the sequence.
    """
    if not yolo_results:
        return {"crowd_mean": 0.0, "crowd_std": 0.0, "crowd_max": 0.0}
    counts = [sum(1 for obj in frame.get("objects", []) if obj.get("class") == "person")
              for frame in yolo_results]
    return {
        "crowd_mean": float(np.mean(counts)),
        "crowd_std": float(np.std(counts)),
        "crowd_max": float(np.max(counts)),
    }


def process_clip(video_path: Path, yolo_path: Path) -> Dict[str, float]:
    """
    Process a single clip to compute environment features.

    Parameters
    ----------
    video_path : Path
        Path to the video file for the clip.
    yolo_path : Path
        Path to the JSON file containing YOLO detection results.

    Returns
    -------
    Dict[str, float]
        A dictionary with aggregated brightness, motion, and crowd features.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if yolo_path.is_file():
        with open(yolo_path, 'r') as f:
            yolo_results = json.load(f)
    else:
        yolo_results = []

    features: Dict[str, float] = {}
    features.update(compute_brightness(frames))
    features.update(compute_motion(frames))
    features.update(compute_crowd(yolo_results))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute environment features from video clips.")
    parser.add_argument('--cfg', required=True, help='Path to dataset configuration YAML file.')
    parser.add_argument('--split', nargs='*', default=['train', 'val', 'test'],
                        help='Dataset splits to process (default: all).')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    # Assume configuration provides directories for clips and YOLO detections
    clips_root = Path(cfg['data']['clips_dir']) if 'clips_dir' in cfg.get('data', {}) else None
    yolo_root = Path(cfg['features']['yolo']) if 'yolo' in cfg.get('features', {}) else None
    stats_dir = Path(cfg['features']['stats']) if 'stats' in cfg.get('features', {}) else Path('features/stats')
    stats_dir.mkdir(parents=True, exist_ok=True)

    if clips_root is None or yolo_root is None:
        raise ValueError("Configuration must specify 'data.clips_dir' and 'features.yolo' paths.")

    all_records: List[Dict[str, Any]] = []
    for split in args.split:
        split_dir = clips_root / split
        if not split_dir.exists():
            print(f"Warning: split directory {split_dir} does not exist; skipping.")
            continue
        for video_file in sorted(split_dir.glob('*.mp4')):
            clip_id = video_file.stem
            yolo_file = yolo_root / split / f"{clip_id}.json"
            features = process_clip(video_file, yolo_file)
            features['clip_id'] = clip_id
            features['split'] = split
            all_records.append(features)
            print(f"Processed {clip_id} in split {split}")

    df = pd.DataFrame(all_records)
    output_path = stats_dir / 'env_features.parquet'
    df.to_parquet(output_path, index=False)
    print(f"Saved environment features to {output_path}")


if __name__ == '__main__':
    main()
