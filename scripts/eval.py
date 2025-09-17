#!/usr/bin/env python
"""
Evaluation script for anomaly premonition detection.
This script computes common metrics such as ROC‑AUC, Average Precision (AP),
and premonition‑specific metrics like Earliness, Time To Detection (TTD),
and False Alarm Rate (FAR) per hour.

Predictions should be provided as a CSV file with columns:
  video: identifier for each video or clip
  time: timestamp in seconds relative to the start of the video/clip
  score: predicted premonition score in [0,1]
  label: binary ground‑truth label (1 within premonition window, 0 otherwise)

The configuration YAML should point to the dataset annotations (ground truth
anomaly intervals) via the 'ann' section if earliness and TTD are to be
computed. Otherwise, only AUC and AP will be reported.

This implementation prioritizes clarity and reproducibility over speed. It
assumes that timestamps are continuous in seconds and that annotations are
provided as lists of start/end times for each video.
"""

import argparse
import json
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def load_config(cfg_path: str) -> dict:
    """Load YAML configuration file."""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def compute_basic_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """Compute ROC‑AUC and Average Precision (AP).

    Returns NaN for metrics that cannot be computed due to lack of positive/negative samples.
    """
    # Ensure arrays are one‑dimensional
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    # AUC and AP require both positive and negative samples
    if len(np.unique(y_true)) < 2:
        return (float('nan'), float('nan'))
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return (auc, ap)


def determine_threshold_at_precision(y_true: np.ndarray, scores: np.ndarray, precision_target: float) -> float:
    """Find a score threshold such that precision is at least precision_target.

    This uses the precision‑recall curve to identify the smallest threshold
    where precision >= precision_target. If no such threshold exists, the
    lowest threshold from the curve is returned. If the curve is empty, 0.5 is used.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision array includes an extra entry corresponding to threshold=0
    # iterate through thresholds to find first precision >= target
    selected_threshold = None
    for p, t in zip(precision, np.append(thresholds, thresholds[-1] if len(thresholds) > 0 else 0.5)):
        if p >= precision_target:
            selected_threshold = t
            break
    if selected_threshold is None:
        # default to the smallest threshold if no precision >= target
        selected_threshold = thresholds[0] if len(thresholds) > 0 else 0.5
    return selected_threshold


def compute_earliness_ttd_far(annotations: list, preds_df: pd.DataFrame, precision_target: float = 0.8) -> tuple:
    """Compute Earliness, Time‑To‑Detection (TTD), and False Alarm Rate (FAR).

    annotations: list of dicts, each with keys 'video', 'start_time', 'end_time'
    preds_df: DataFrame with columns ['video','time','score','label']
    precision_target: precision level for determining the detection threshold

    Returns a tuple (earliness_median, ttd_median, far_per_hour). If there are
    no positive events, NaN values are returned for earliness and TTD, and FAR
    is computed based on false positives.
    """
    # Compute global threshold at desired precision
    y_true = preds_df['label'].values
    scores = preds_df['score'].values
    threshold = determine_threshold_at_precision(y_true, scores, precision_target)
    # Precompute detection times for each video
    detections = {}
    for video, group in preds_df.groupby('video'):
        # sort by time to find earliest detection
        group_sorted = group.sort_values('time')
        det_times = group_sorted[group_sorted['score'] >= threshold]['time'].values
        detections[video] = det_times[0] if len(det_times) > 0 else None
    earliness_list = []
    ttd_list = []
    # Evaluate each annotated anomaly interval
    for ann in annotations:
        video = ann['video']
        start = ann['start_time']
        det_time = detections.get(video)
        if det_time is not None:
            earliness_list.append(start - det_time)
            ttd_list.append(det_time - start)
    # Compute FAR: count predictions above threshold outside any annotated premonition window
    # Mark times that fall within any annotation as valid
    # Build boolean mask for each prediction
    is_false_alarm = []
    for idx, row in preds_df.iterrows():
        if row['score'] < threshold:
            is_false_alarm.append(False)
            continue
        # Check if this positive prediction time falls within any annotated window
        video_annotations = [a for a in annotations if a['video'] == row['video']]
        in_window = False
        for ann in video_annotations:
            if ann['start_time'] <= row['time'] < ann['end_time']:
                in_window = True
                break
        is_false_alarm.append(not in_window)
    false_positives = sum(is_false_alarm)
    # Estimate total duration of predictions across all videos (in seconds)
    # Use max time minus min time per video and sum durations
    total_seconds = 0.0
    for video, group in preds_df.groupby('video'):
        times = group['time'].values
        if len(times) > 0:
            total_seconds += (times.max() - times.min())
    far_per_hour = (false_positives / (total_seconds / 3600.0)) if total_seconds > 0 else float('nan')
    # Compute medians
    earliness_median = float('nan') if len(earliness_list) == 0 else float(np.median(earliness_list))
    ttd_median = float('nan') if len(ttd_list) == 0 else float(np.median(ttd_list))
    return (earliness_median, ttd_median, far_per_hour)


def load_annotations(ann_path: str) -> list:
    """Load anomaly annotations from a JSON file.

    The JSON should be a dict mapping video identifiers to a list of intervals,
    each with 'start' and 'end' keys. Returns a list of dicts with keys
    'video', 'start_time', and 'end_time'.
    """
    annotations = []
    with open(ann_path, 'r') as f:
        data = json.load(f)
    for video, intervals in data.items():
        for interval in intervals:
            start = interval.get('start')
            end = interval.get('end')
            if start is not None and end is not None:
                annotations.append({'video': video, 'start_time': float(start), 'end_time': float(end)})
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly premonition predictions.")
    parser.add_argument('--cfg', type=str, required=True, help='Path to evaluation config YAML.')
    parser.add_argument('--preds', type=str, required=True, help='CSV file containing predictions.')
    parser.add_argument('--out_csv', type=str, default=None, help='Optional CSV file to save results.')
    parser.add_argument('--precision', type=float, default=0.8, help='Precision target for earliness/TTD calculation.')
    args = parser.parse_args()
    # Load configuration
    cfg = load_config(args.cfg)
    # Load predictions CSV
    preds_df = pd.read_csv(args.preds)
    # Compute basic metrics
    auc, ap = compute_basic_metrics(preds_df['label'].values, preds_df['score'].values)
    # Compute earliness, TTD, FAR if annotations available
    ann_path = None
    if isinstance(cfg.get('ann'), dict):
        # expected key 'gt_file' or similar
        ann_path = cfg['ann'].get('gt_file') or cfg['ann'].get('ground_truth')
    earliness = float('nan')
    ttd = float('nan')
    far = float('nan')
    if ann_path is not None:
        annotations = load_annotations(ann_path)
        if len(annotations) > 0:
            earliness, ttd, far = compute_earliness_ttd_far(annotations, preds_df, precision_target=args.precision)
    # Prepare results
    results = {
        'AUC': auc,
        'AP': ap,
        'Earliness': earliness,
        'TTD': ttd,
        'FAR_per_hour': far,
    }
    # Print to stdout
    for k, v in results.items():
        print(f"{k}: {v}")
    # Save to CSV if requested
    if args.out_csv:
        out_rows = [[k, v] for k, v in results.items()]
        out_df = pd.DataFrame(out_rows, columns=['metric', 'value'])
        out_df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
