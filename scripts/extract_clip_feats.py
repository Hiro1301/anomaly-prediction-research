#!/usr/bin/env python3
"""
Extract CLIP embeddings for each video clip.

This script loads a pre-trained CLIP model and processes video clips to
compute frame-level embeddings. The embeddings are aggregated (mean and max)
across frames to form a clip-level representation and saved as .npy files
under the features/clip directory specified in the configuration.

To extend this script, you can adjust the sampling strategy (e.g. using
different numbers of frames), change the CLIP model variant, or include
other aggregation functions.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import cv2
from PIL import Image
import yaml
from transformers import CLIPProcessor, CLIPModel


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def extract_clip_embeddings(video_path: Path, model: CLIPModel, processor: CLIPProcessor,
                            clip_len: int = 16) -> np.ndarray:
    """
    Extract frame-level CLIP embeddings from a video and return an array of shape (N, D),
    where N is the number of sampled frames and D is the embedding dimension.

    Parameters
    ----------
    video_path : Path
        Path to the video file to process.
    model : CLIPModel
        Pretrained CLIP model.
    processor : CLIPProcessor
        Preprocessing pipeline for CLIP.
    clip_len : int, optional
        Number of frames to sample from the video, by default 16.

    Returns
    -------
    np.ndarray
        Array of frame embeddings.
    """
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return np.zeros((0, model.config.projection_dim), dtype=np.float32)
    # Sample evenly spaced indices
    indices = np.linspace(0, max(total_frames - 1, 0), num=clip_len, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    if not frames:
        return np.zeros((0, model.config.projection_dim), dtype=np.float32)
    inputs = processor(images=frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings.cpu().numpy()
    return embeddings


def aggregate_embeddings(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Aggregate frame-level embeddings into summary statistics."""
    if embeddings.size == 0:
        dim = 512  # Default dimension for CLIP ViT-B/32
        return {
            'mean': np.zeros(dim, dtype=np.float32),
            'max': np.zeros(dim, dtype=np.float32),
        }
    return {
        'mean': embeddings.mean(axis=0),
        'max': embeddings.max(axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP features from video clips.")
    parser.add_argument('--cfg', required=True, help='Path to dataset configuration YAML.')
    parser.add_argument('--split', nargs='*', default=['train', 'val', 'test'],
                        help='Dataset splits to process (default: all).')
    parser.add_argument('--model', default='openai/clip-vit-base-patch32',
                        help='HuggingFace model id for the CLIP variant.')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    clips_root = Path(cfg['data']['clips_dir'])
    clip_len = cfg['data'].get('clip_len', 16)
    out_dir = Path(cfg['features']['clip'])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    for split in args.split:
        split_dir = clips_root / split
        out_split_dir = out_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        if not split_dir.exists():
            print(f"Warning: split directory {split_dir} does not exist; skipping.")
            continue
        for video_file in sorted(split_dir.glob('*.mp4')):
            clip_id = video_file.stem
            embeddings = extract_clip_embeddings(video_file, model, processor, clip_len)
            agg = aggregate_embeddings(embeddings)
            # Save both raw embeddings and aggregated embeddings as needed
            np.save(out_split_dir / f"{clip_id}_frames.npy", embeddings)
            np.save(out_split_dir / f"{clip_id}_mean.npy", agg['mean'])
            np.save(out_split_dir / f"{clip_id}_max.npy", agg['max'])
            print(f"Processed {clip_id} ({split})")


if __name__ == '__main__':
    main()
