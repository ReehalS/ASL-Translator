"""Ensemble deeper per-window model and MIL pooled model.

Loads:
  - Models/gesture_wlasl_deeper.keras (per-window classifier)
  - Models/gesture_wlasl_mil_long_mil.h5 (MIL pooled model)

The script prepares bags using the same logic as train_tf_mil, then computes
predictions from each model and averages probabilities.
"""
import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
sys.path.insert(0, os.getcwd())

from Code.dataset.windowed_generator import generate_windows_from_csv
import argparse
import tensorflow as tf
import os


def prepare_bags(csv_path, window_size=16, stride=4, bag_size=32):
    bags = defaultdict(list)
    labels = {}
    for item in generate_windows_from_csv(csv_path, window_size=window_size, stride=stride, pad=True):
        bags[item['clip_id']].append(item['window'])
        labels[item['clip_id']] = item['label']
    bag_list = []
    label_list = []
    for clip_id, wins in bags.items():
        arr = np.stack(wins, axis=0)
        # sample/pad to bag_size
        if arr.shape[0] >= bag_size:
            idxs = np.random.choice(arr.shape[0], bag_size, replace=False)
            arr = arr[idxs]
        else:
            repeats = bag_size - arr.shape[0]
            pad = np.repeat(arr[-1:], repeats, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        bag_list.append(arr)
        label_list.append(labels[clip_id])
    return bag_list, label_list


def load_model_fallback(path):
    # try load as full model, then try loading weights into a new model if caller builds one
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        # not a full model or contains custom layers; return None to signal fallback
        return None


def main():
    parser = argparse.ArgumentParser(description='Ensemble MIL and deeper models')
    parser.add_argument('--csv', default=os.path.join('Dataset', 'Generated_Data', 'wlasl_pipeline_frames.csv'))
    parser.add_argument('--deeper', default=os.path.join('Models', 'gesture_wlasl_deeper.keras'))
    parser.add_argument('--mil', default=os.path.join('Models', 'gesture_wlasl_mil_finetuned_mil.keras'))
    parser.add_argument('--deeper_labels', default=os.path.join('Models', 'gesture_wlasl_deeper_labels.json'))
    parser.add_argument('--mil_labels', default=os.path.join('Models', 'gesture_wlasl_mil_finetuned_mil_labels.json'))
    parser.add_argument('--bag_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    csv = args.csv
    deeper_path = args.deeper
    mil_path = args.mil
    deeper = load_model_fallback(deeper_path)
    # rebuild MIL architecture to avoid Lambda deserialization issues if needed
    from Code.train_tf_mil import build_encoder
    # load label maps
    with open(args.deeper_labels) as f:
        deeper_labels = json.load(f)
    with open(args.mil_labels) as f:
        mil_labels = json.load(f)
    # invert label maps
    deeper_inv = {int(v):k for k,v in deeper_labels.items()}
    mil_inv = {int(v):k for k,v in mil_labels.items()}

    bag_list, label_list = prepare_bags(csv, window_size=16, stride=4, bag_size=args.bag_size)
    feat = bag_list[0].shape[2]

    # build MIL model structure matching train_tf_mil
    encoder = build_encoder((16, feat), embed_dim=128)
    bag_input = tf.keras.layers.Input(shape=(None, 16, feat), dtype=tf.float32)
    td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
    att_dense = tf.keras.layers.Dense(1)(td)
    att = tf.keras.layers.Softmax(axis=1)(att_dense)
    pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
    pooled = tf.keras.layers.Reshape((128,))(pooled)
    out = tf.keras.layers.Dense(128, activation='relu')(pooled)
    out = tf.keras.layers.Dense(len(mil_labels), activation='softmax')(out)
    mil = tf.keras.Model(bag_input, out)
    # attempt to load MIL as full model first, else load weights
    mil_full = load_model_fallback(mil_path)
    if mil_full is not None:
        mil = mil_full
    else:
        # try loading weights into the rebuilt MIL
        try:
            mil.load_weights(mil_path)
        except Exception:
            # try loading .weights.h5 companion
            wpath = mil_path
            if not wpath.endswith('.weights.h5') and os.path.exists(mil_path + '.weights.h5'):
                wpath = mil_path + '.weights.h5'
            mil.load_weights(wpath)
    # split 80/20 same as training
    N = len(bag_list)
    idx = np.arange(N)
    np.random.seed(args.seed)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    val_idx = idx[split:]

    # Build per-window predictor from deeper model: assume deeper accepts (window, feats)
    # If deeper was trained with input shape (16, feat), we can run it on each window
    correct = 0
    total = 0
    for j in val_idx:
        bag = bag_list[j]  # shape (bag_size, window, feat)
        # deeper expects (batch, window, feat) per-window; so run on bag windows
        # produce probs per window, then average to clip-level
        bs = bag.shape[0]
        windows = bag.reshape((bs, bag.shape[1], bag.shape[2]))
        # if deeper is None (failed to load), try loading as full model now
        if deeper is None:
            deeper = load_model_fallback(deeper_path)
            if deeper is None:
                raise RuntimeError(f'Failed to load deeper model from {deeper_path}')
        deeper_probs = deeper.predict(windows, verbose=0)
        deeper_clip_prob = deeper_probs.mean(axis=0)

        # mil expects input shape (1, num_windows, window, feat)
        mil_input = np.expand_dims(bag, axis=0)
        mil_probs = mil.predict(mil_input, verbose=0)[0]

        # Align label spaces: create arrays of length union of labels using names
        # Build mapping from deeper label index to mil index
        # Actually deeper_labels and mil_labels map label_name->idx
        deeper_name_to_idx = deeper_labels
        mil_name_to_idx = mil_labels

        # compute probability vector on mil label space by mapping deeper_clip_prob
        mil_from_deeper = np.zeros(len(mil_name_to_idx), dtype=float)
        for name, di in deeper_name_to_idx.items():
            di = int(di)
            if name in mil_name_to_idx:
                mi = int(mil_name_to_idx[name])
                mil_from_deeper[mi] = deeper_clip_prob[di]

        # now mil_probs is already on mil label indices
        # average probabilities (on mil label space)
        avg_prob = (mil_from_deeper + mil_probs) / 2.0
        pred = int(np.argmax(avg_prob))
        true_name = label_list[j]
        true_idx = int(mil_name_to_idx[true_name])
        if pred == true_idx:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    print('Ensemble val acc', acc, ' (correct', correct, 'of', total, ')')


if __name__ == '__main__':
    main()
