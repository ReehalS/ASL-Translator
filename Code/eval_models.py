"""Evaluate MIL and deeper models on the project's validation split.

Usage examples (from repo root):
  python Code/eval_models.py --mil Models/gesture_wlasl_mil_finetuned_mil.h5
  python Code/eval_models.py --deeper Models/gesture_wlasl_deeper_aug.keras --jitter 2 --lm_noise 0.02
"""
import os
import json
import numpy as np
import tensorflow as tf
import argparse

sys_path = os.getcwd()
import sys
sys.path.insert(0, sys_path)

from Code.pretrain_and_finetune_mil import prepare_bags
from Code.dataset.windowed_generator import windows_to_numpy


def eval_mil(model_path, csv_path, bag_size=32, batch=8):
    # deterministic bag sampling
    np.random.seed(0)
    bag_list, label_list = prepare_bags(csv_path, window_size=16, stride=4, bag_size=bag_size)
    if not bag_list:
        return 0.0
    # build label map as saved by training (sorted uniq)
    uniq = sorted(set(label_list))
    lm = {v:i for i,v in enumerate(uniq)}

    # Try loading full model first; fallback to reconstruct+load_weights if Lambda deserialization fails
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # Try reconstructing exact architecture using project's encoder and loading weights file if available
        from Code.train_tf_mil import build_encoder
        feat = bag_list[0].shape[2]
        embed = 128
        # build encoder and MIL with fixed bag dimension = bag_size used in prepare_bags
        bag_size = bag_list[0].shape[0]
        bag_input = tf.keras.layers.Input(shape=(bag_size, bag_list[0].shape[1], feat), dtype=tf.float32)
        encoder = build_encoder((bag_list[0].shape[1], feat), embed_dim=embed)
        td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
        att_dense = tf.keras.layers.Dense(1)(td)
        att = tf.keras.layers.Softmax(axis=1)(att_dense)
        pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
        pooled = tf.keras.layers.Reshape((embed,))(pooled)
        out = tf.keras.layers.Dense(128, activation='relu')(pooled)
        out = tf.keras.layers.Dense(len(uniq), activation='softmax')(out)
        model = tf.keras.Model(bag_input, out)
        # try to load weights from a sibling .weights.h5 file (saved by training)
        weight_path = None
        if model_path.endswith('.keras'):
            alt = model_path.replace('.keras', '.weights.h5')
            if os.path.exists(alt):
                weight_path = alt
        if weight_path is None and os.path.exists(model_path):
            # if provided path is actually an h5 weights file, try that
            weight_path = model_path
        if weight_path and os.path.exists(weight_path):
            try:
                model.load_weights(weight_path, by_name=True)
            except Exception:
                try:
                    model.load_weights(weight_path)
                except Exception:
                    print('Warning: failed to load MIL weights; skipping eval for', model_path)
                    return 0.0
        else:
            print('Warning: no weights file found for', model_path)
            return 0.0

    # create batches similar to training
    N = len(bag_list)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    val_idx = idx[split:]

    correct = 0
    total = 0
    for i in range(0, len(val_idx), batch):
        batch_idx = val_idx[i:i+batch]
        max_wins = max(bag_list[j].shape[0] for j in batch_idx)
        feat = bag_list[0].shape[2]
        batch_bags = np.zeros((len(batch_idx), max_wins, bag_list[0].shape[1], feat), dtype=np.float32)
        batch_labels = np.zeros((len(batch_idx),), dtype=np.int32)
        for ii, j in enumerate(batch_idx):
            wins = bag_list[j]
            batch_bags[ii, :wins.shape[0]] = wins
            batch_labels[ii] = lm[label_list[j]]
        preds = model.predict(batch_bags)
        preds_idx = preds.argmax(axis=1)
        correct += int((preds_idx == batch_labels).sum())
        total += len(batch_idx)
    acc = correct / total if total else 0.0
    return acc


def eval_deeper(model_path, labels_json, csv_path, jitter=0, lm_noise=0.0):
    X, y, clips = windows_to_numpy(csv_path, window_size=16, stride=4, jitter=jitter, lm_noise=lm_noise)
    if X.size == 0:
        return 0.0
    with open(labels_json, 'r') as f:
        lm = json.load(f)
    # recreate clip-wise split used in training
    from collections import defaultdict
    clip_to_idx = defaultdict(list)
    for i, c in enumerate(clips):
        clip_to_idx[c].append(i)
    clip_ids = list(clip_to_idx.keys())
    split = int(0.8 * len(clip_ids))
    train_clips = set(clip_ids[:split])
    val_idx = [i for i, c in enumerate(clips) if c not in train_clips]

    X_val = X[val_idx]
    y_val = np.array([lm[v] for v in y[val_idx]])
    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_val)
    preds_idx = preds.argmax(axis=1)
    acc = float((preds_idx == y_val).mean())
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mil', help='Path to MIL model (.h5 or .keras)')
    parser.add_argument('--deeper', help='Path to deeper model (.keras)')
    parser.add_argument('--labels', help='Labels json for deeper model')
    parser.add_argument('--jitter', type=int, default=0)
    parser.add_argument('--lm_noise', type=float, default=0.0)
    parser.add_argument('--csv', default=os.path.join('Dataset', 'Generated_Data', 'wlasl_pipeline_frames.csv'))
    args = parser.parse_args()

    if args.mil:
        acc = eval_mil(args.mil, args.csv)
        print('MIL val acc', acc)
    if args.deeper and args.labels:
        acc = eval_deeper(args.deeper, args.labels, args.csv, jitter=args.jitter, lm_noise=args.lm_noise)
        print('Deeper val acc', acc)


if __name__ == '__main__':
    main()
