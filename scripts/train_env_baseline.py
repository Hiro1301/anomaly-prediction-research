#!/usr/bin/env python3
"""
Train a baseline classifier using environment features for premonitory anomaly detection.

This script loads aggregated environment features (brightness, motion, crowd statistics)
and corresponding premonition labels for a specified temporal window (Delta t) and trains
an XGBoost classifier. It evaluates the classifier on the validation and test splits and
reports ROC-AUC and Average Precision (AP). The trained model is saved to disk for later use.

This is a minimal baseline implementation; more sophisticated models (e.g. small
Transformers) can be incorporated by extending this script.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import yaml


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train environment-based anomaly premonition baseline.")
    parser.add_argument('--cfg', required=True, help='Path to dataset configuration YAML.')
    parser.add_argument('--delta', type=int, default=10, help='Delta t (seconds) for premonition labels.')
    parser.add_argument('--model', default='xgboost', choices=['xgboost'], help='Model type to train.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    # Paths for features and labels
    stats_dir = Path(cfg['features']['stats'])
    features_file = stats_dir / 'env_features.parquet'
    labels_dir = Path(cfg['annotations']['premonition_dir'])
    labels_file = labels_dir / f'premonition_{args.delta}s.csv'
    models_dir = Path(cfg.get('models', {}).get('context', 'models/context'))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not features_file.is_file():
        raise FileNotFoundError(f"Environment feature file not found: {features_file}")
    if not labels_file.is_file():
        raise FileNotFoundError(f"Premonition label file not found: {labels_file}")
    df_features = pd.read_parquet(features_file)
    df_labels = pd.read_csv(labels_file)

    # Align column names
    if 'video' in df_labels.columns:
        df_labels = df_labels.rename(columns={'video': 'clip_id'})
    if 'label' not in df_labels.columns and 'premonition' in df_labels.columns:
        df_labels = df_labels.rename(columns={'premonition': 'label'})
    # Merge features into label dataframe
    df = df_labels.merge(df_features, on='clip_id', how='left', suffixes=('', '_feat'))

    # Drop rows without features
    df = df.dropna(subset=['brightness_mean', 'crowd_mean', 'motion_mean'], how='any')

    # Prepare feature matrix X and target y
    drop_cols = {'clip_id', 'split', 'time', 'timestamp', 'label'}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Identify splits if present
    if 'split' in df.columns:
        train_mask = df['split'] == 'train'
        val_mask = df['split'] == 'val'
        test_mask = df['split'] == 'test'
    else:
        # If no split column, perform a simple random split
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=args.seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)
        train_mask = val_mask = test_mask = None

    # Train model
    if args.model == 'xgboost':
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=args.seed,
            n_jobs=4,
        )
        if train_mask is not None:
            model.fit(X[train_mask], y[train_mask],
                      eval_set=[(X[val_mask], y[val_mask])],
                      verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # Evaluate on test split
    if train_mask is not None:
        prob = model.predict_proba(X[test_mask])[:, 1]
        y_true = y[test_mask]
    else:
        prob = model.predict_proba(X_test)[:, 1]
        y_true = y_test

    auc = roc_auc_score(y_true, prob)
    ap = average_precision_score(y_true, prob)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test Average Precision: {ap:.4f}")

    # Save model
    model_path = models_dir / f'env_baseline_xgb_{args.delta}s.json'
    model.save_model(str(model_path))
    scaler_path = models_dir / f'env_baseline_scaler_{args.delta}s.pkl'
    # Save scaler using joblib
    try:
        import joblib
        joblib.dump(scaler, scaler_path)
    except ImportError:
        # Fallback: save as numpy array
        np.save(scaler_path.with_suffix('.npy'), scaler.scale_)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
