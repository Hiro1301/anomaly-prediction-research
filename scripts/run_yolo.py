#!/usr/bin/env python3
"""
Run YOLO object detection on video clips to generate detection results.

This script uses the Ultralytics YOLOv8 implementation to detect objects
such as persons, bicycles, cars, bags, and tools in each frame of a video
clip. The detection results for each frame are stored as a JSON list,
where each frame entry contains a list of detected objects with their
class names, confidence scores, and bounding boxes.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import yaml

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # placeholder if ultralytics is not installed


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def run_detection_on_video(video_path: Path, model: Any, class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Run object detection on a single video and return structured results.

    Parameters
    ----------
    video_path : Path
        Path to the input video clip (.mp4).
    model : YOLO
        Pre-loaded YOLO model from ultralytics.
    class_names : List[str]
        Classes to keep from detection results.

    Returns
    -------
    List[Dict[str, Any]]
        A list where each item corresponds to a frame and contains a
        dictionary with the frame index and a list of detected objects.
    """
    results = []
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLO prediction
        if model is None:
            raise ImportError("Ultralytics YOLO is not installed. Please install ultralytics to use this script.")
        detections = model(frame)  # returns a list of results
        frame_objs = []
        for det in detections:
            boxes = det.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                name = det.names.get(cls_id, str(cls_id))
                if name not in class_names:
                    continue
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                frame_objs.append({
                    "class": name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                })
        results.append({"frame": frame_idx, "objects": frame_objs})
        frame_idx += 1
    cap.release()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on video clips.")
    parser.add_argument('--cfg', required=True, help='Path to dataset configuration YAML.')
    parser.add_argument('--classes', nargs='*', default=['person', 'bicycle', 'car', 'bag', 'tool'],
                        help='Object classes to retain in the output.')
    parser.add_argument('--model', default='yolov8x.pt',
                        help='YOLO model weights (e.g. yolov8n.pt, yolov8x.pt).')
    parser.add_argument('--split', nargs='*', default=['train', 'val', 'test'],
                        help='Dataset splits to process.')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    clips_root = Path(cfg['data']['clips_dir'])
    out_root = Path(cfg['features']['yolo'])
    out_root.mkdir(parents=True, exist_ok=True)

    if YOLO is None:
        raise ImportError("Ultralytics YOLO is not available. Install it via pip install ultralytics")
    model = YOLO(args.model)

    for split in args.split:
        split_dir = clips_root / split
        out_split_dir = out_root / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Skipping split {split}.")
            continue
        for video_file in sorted(split_dir.glob('*.mp4')):
            clip_id = video_file.stem
            detections = run_detection_on_video(video_file, model, args.classes)
            out_path = out_split_dir / f"{clip_id}.json"
            with open(out_path, 'w') as f:
                json.dump(detections, f)
            print(f"Processed {clip_id} ({split})")


if __name__ == '__main__':
    main()
