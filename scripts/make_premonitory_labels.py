#!/usr/bin/env python3
"""
Generate premonitory labels for anomaly premonition detection.

For each anomaly in the ground truth annotation file, this script marks the time window
[T - Δt, T) preceding the anomaly start time T as 'premonitory' (label=1) and all other times as 0.
Multiple Δt values can be specified to generate separate label files.

The input ground truth should be a JSON file with a list of anomalies of the form:
[
  {
    "video": "04",
    "start_time": 120.5,
    "end_time": 135.2
  },
  ...
]

Output files are saved in the directory specified by the configuration under
`annotations.premonition_dir` with filenames like `premonition_10s.csv`.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yaml


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_labels(annotations: List[Dict[str, Any]], deltas: List[int], output_dir: str, fps: int, clip_len: int) -> None:
    """
    Generate premonitory labels for each delta.
    Args:
        annotations: list of anomaly annotations with keys 'video', 'start_time', 'end_time'.
        deltas: list of delta values in seconds.
        output_dir: directory to save the generated CSV files.
        fps: frames per second of the dataset.
        clip_len: length of a clip in frames (used to compute time step resolution).
    """
    os.makedirs(output_dir, exist_ok=True)
    time_step = clip_len / fps
    # Build a per-video list of anomaly start times
    starts_by_video: Dict[str, List[float]] = {}
    for entry in annotations:
        starts_by_video.setdefault(entry["video"], []).append(entry["start_time"])
    # For each delta, generate labels
    for delta in deltas:
        rows = []
        for video, starts in starts_by_video.items():
            max_start = max(starts)
            # Determine the timeline range up to the latest anomaly start
            t = 0.0
            while t <= max_start:
                # Determine if t is within any premonitory window
                label = 0
                for start_time in starts:
                    if start_time - delta <= t < start_time:
                        label = 1
                        break
                rows.append({"video": video, "time": round(t, 3), "label": label})
                t += time_step
        df = pd.DataFrame(rows)
        out_path = Path(output_dir) / f"premonition_{delta}s.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved labels for Δt={delta}s to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate premonitory labels from ground truth annotations.")
    parser.add_argument("--cfg", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--deltas", type=int, nargs="+", default=[10], help="List of Δt values in seconds.")
    args = parser.parse_args()

    config = load_config(args.cfg)
    ann_cfg = config.get("annotations", {})
    gt_path = ann_cfg.get("ground_truth")
    if not gt_path:
        raise ValueError("Ground truth path not found in configuration under annotations.ground_truth")
    with open(gt_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    output_dir = ann_cfg.get("premonition_dir", "ann/premonition_labels")
    fps = config.get("fps", 15)
    clip_len = config.get("clip_len", 16)

    generate_labels(annotations, args.deltas, output_dir, fps, clip_len)


if __name__ == "__main__":
    main()
