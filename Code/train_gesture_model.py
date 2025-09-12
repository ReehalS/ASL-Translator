"""Train a simple gesture classifier using aggregated mean+std features or flattened sequences.

This script expects a CSV with columns: x0,y0,...,x20,y20,clip_id,label
The converter `Code/convert_gesture_dataset.py` can produce such a CSV.

Usage examples:
  python Code/train_gesture_model.py --input Dataset/Generated_Data/gestures.csv --mode aggregate --out Models/gesture_mlp_agg.pkl
  python Code/train_gesture_model.py --input Dataset/Generated_Data/gestures.csv --mode flatten --seq-len 8 --out Models/gesture_mlp_flat.pkl

Modes:
 - aggregate: group frames by clip_id, compute mean and std per coordinate -> 84 features
 - flatten: pad/truncate frames per clip to seq_len and flatten -> 42*seq_len features

Saves a scikit-learn MLPClassifier using joblib and prints a classification report.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_csv(path):
    df = pd.read_csv(path)
    return df


def aggregate_by_clip(df):
    # group by clip_id and label
    cols = [c for c in df.columns if c.startswith('x') or c.startswith('y')]
    agg_rows = []
    for (clip_id, label), g in df.groupby(['clip_id', 'label']):
        arr = g[cols].values.astype(float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        feat = np.concatenate([mean, std])
        agg_rows.append((clip_id, label, feat))
    ids = [r[0] for r in agg_rows]
    labels = [r[1] for r in agg_rows]
    X = np.vstack([r[2] for r in agg_rows])
    return X, labels, ids


def flatten_by_clip(df, seq_len):
    cols = [c for c in df.columns if c.startswith('x') or c.startswith('y')]
    clips = []
    labels = []
    ids = []
    for (clip_id, label), g in df.groupby(['clip_id', 'label']):
        arr = g[cols].values.astype(float)
        # pad or truncate to seq_len
        if arr.shape[0] >= seq_len:
            sub = arr[:seq_len]
        else:
            # pad by repeating last frame
            pad = np.repeat(arr[-1:], seq_len - arr.shape[0], axis=0)
            sub = np.vstack([arr, pad])
        feat = sub.reshape(-1)
        clips.append(feat)
        labels.append(label)
        ids.append(clip_id)
    X = np.vstack(clips)
    return X, labels, ids


def build_and_train(X, y, out_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Confusion matrix shape:", cm.shape)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Saved model to {out_path}")
    return clf, (X_test, y_test, y_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--mode", choices=["aggregate", "flatten"], default="aggregate")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = load_csv(args.input)
    if args.mode == "aggregate":
        X, labels, ids = aggregate_by_clip(df)
    else:
        X, labels, ids = flatten_by_clip(df, args.seq_len)

    clf, test_info = build_and_train(X, labels, args.out)
    X_test, y_test, y_pred = test_info


if __name__ == "__main__":
    main()
