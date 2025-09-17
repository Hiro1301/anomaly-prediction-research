#!/usr/bin/env python
"""
Download dataset files for anomaly detection research.

This script downloads and extracts datasets used in the anomaly premonition project.
It supports UCSD Ped2, UCSD Ped1, Avenue, ShanghaiTech, NWPU Campus, and CUVA datasets.
The download URLs provided here are for convenience; please verify and update them
according to the dataset's official distribution page. Some datasets may require
manual download due to license restrictions.

Usage:
    python download_dataset.py --dataset ucsd_ped2 --output_dir data/

Example:
    python download_dataset.py --dataset shanghaitech --output_dir data/ShanghaiTech
"""

import argparse
import os
import sys
import requests
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import tarfile
import zipfile

# Mapping from dataset name to download URL
DATASET_URLS: Dict[str, str] = {
    "ucsd_ped2": "http://www.svcl.ucsd.edu/projects/anomaly/dataset/UCSD_Anomaly_Dataset.tar.gz",
    "ucsd_ped1": "http://www.svcl.ucsd.edu/projects/anomaly/dataset/UCSD_Anomaly_Dataset.tar.gz",
    # The Avenue dataset is often distributed via research groups; update this URL as needed.
    "avenue": "https://data.vision.ee.ethz.ch/cvl/avenue/Avenue_Dataset.zip",
    # ShanghaiTech dataset:
    "shanghaitech": "https://download.oracle.com/otn/sample_code/ucsd/ShanghaiTech.zip",
    # NWPU Campus dataset (placeholder URL; please replace with official link):
    "nwpu": "https://download.example.com/nwpu_campus_anomaly_dataset.zip",
    # CUVA dataset (placeholder URL; please replace with official link):
    "cuva": "https://download.example.com/cuva_anomaly_dataset.zip",
}

def download_file(url: str, dest_path: Path) -> None:
    """Download a file from a URL to a local destination with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        with open(dest_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def extract_archive(file_path: Path, output_dir: Path) -> None:
    """Extract tar.gz or zip archives to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Determine archive type
    suffixes = ''.join(file_path.suffixes)
    if suffixes.endswith('.tar.gz') or suffixes.endswith('.tgz'):
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
    elif suffixes.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print(f"Unsupported archive format for {file_path}", file=sys.stderr)

def download_dataset(dataset: str, output_dir: Path) -> None:
    """Download and extract the specified dataset."""
    dataset = dataset.lower()
    if dataset not in DATASET_URLS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASET_URLS.keys())}")
    url = DATASET_URLS[dataset]
    file_name = url.split('/')[-1]
    dest_file = output_dir / file_name
    print(f"Downloading {dataset} from {url} ...")
    download_file(url, dest_file)
    print(f"Extracting {dest_file} ...")
    extract_archive(dest_file, output_dir)
    print(f"Done. Files extracted to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract datasets for anomaly detection research.")
    parser.add_argument('--dataset', type=str, required=True, help=f"Dataset to download. Options: {list(DATASET_URLS.keys())}")
    parser.add_argument('--output_dir', type=str, default='data', help="Destination directory for downloaded data.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    download_dataset(args.dataset, output_dir)

if __name__ == '__main__':
    main()
