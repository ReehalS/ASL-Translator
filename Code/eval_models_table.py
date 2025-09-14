"""Evaluate MIL, deeper, and ensemble; supports bag sampling repeats and returns per-model accuracies.

Usage: python Code/eval_models_table.py --mil <mil.keras> --deeper <deeper.keras> --mil_labels <...> --deeper_labels <...> --bag_size 32 --seed 0 --sample_runs 1
"""
import os
import json
import numpy as np
import argparse
import tensorflow as tf
from collections import defaultdict
import sys
sys.path.insert(0, os.getcwd())
from Code.dataset.windowed_generator import generate_windows_from_csv
from Code.train_tf_mil import build_encoder


def prepare_bags(csv_path, window_size=16, stride=4, bag_size=32, seed=0, sample_runs=1):
    # returns list of bag_lists where each element is bag_list for a sampling run
    np.random.seed(seed)
    bags = defaultdict(list)
    labels = {}
    for item in generate_windows_from_csv(csv_path, window_size=window_size, stride=stride, pad=True):
        bags[item['clip_id']].append(item['window'])
        labels[item['clip_id']] = item['label']
    clip_ids = list(bags.keys())
    bag_runs = []
    for r in range(sample_runs):
        bag_list = []
        label_list = []
        for clip_id in clip_ids:
            wins = np.stack(bags[clip_id], axis=0)
            if bag_size is not None:
                if wins.shape[0] >= bag_size:
                    idxs = np.random.choice(wins.shape[0], bag_size, replace=False)
                    arr = wins[idxs]
                else:
                    repeats = bag_size - wins.shape[0]
                    pad = np.repeat(wins[-1:], repeats, axis=0)
                    arr = np.concatenate([wins, pad], axis=0)
            else:
                arr = wins
            bag_list.append(arr)
            label_list.append(labels[clip_id])
        bag_runs.append((bag_list, label_list))
    return bag_runs


def load_model_fallback(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


def evaluate(mil_path, deeper_path, mil_labels_path, deeper_labels_path, csv, bag_size=32, seed=0, sample_runs=1):
    # prepare bag runs
    bag_runs = prepare_bags(csv, bag_size=bag_size, seed=seed, sample_runs=sample_runs)
    # load labels
    with open(mil_labels_path) as f:
        mil_labels = json.load(f)
    with open(deeper_labels_path) as f:
        deeper_labels = json.load(f)
    mil_name_to_idx = mil_labels
    deeper_name_to_idx = deeper_labels

    # load models
    deeper = load_model_fallback(deeper_path)
    mil = load_model_fallback(mil_path)
    # build MIL architecture if mil couldn't be loaded
    if mil is None:
        # need feat dim
        feat = bag_runs[0][0][0].shape[2]
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
        # try loading weights
        try:
            mil.load_weights(mil_path)
        except Exception:
            if os.path.exists(mil_path + '.weights.h5'):
                mil.load_weights(mil_path + '.weights.h5')

    # evaluate: for each clip, for each sample_run produce predictions then average across runs if sample_runs>1
    N = len(bag_runs[0][0])
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    val_idx = idx[split:]

    mil_correct = 0
    deeper_correct = 0
    ensemble_correct = 0
    total = 0

    for j in val_idx:
        # aggregate probs across sample_runs
        mil_probs_accum = None
        deeper_probs_accum = None
        for (bag_list, label_list) in bag_runs:
            bag = bag_list[j]
            # deeper per-window
            bs = bag.shape[0]
            windows = bag.reshape((bs, bag.shape[1], bag.shape[2]))
            if deeper is None:
                raise RuntimeError('Deeper model not loaded')
            dprobs = deeper.predict(windows, verbose=0).mean(axis=0)
            m_input = np.expand_dims(bag, axis=0)
            mprobs = mil.predict(m_input, verbose=0)[0]
            if mil_probs_accum is None:
                mil_probs_accum = mprobs
                deeper_probs_accum = dprobs
            else:
                mil_probs_accum += mprobs
                deeper_probs_accum += dprobs
        # average
        mil_probs = mil_probs_accum / float(len(bag_runs))
        deeper_probs = deeper_probs_accum / float(len(bag_runs))

        # map deeper probs to mil label space
        mil_from_deeper = np.zeros(len(mil_name_to_idx), dtype=float)
        for name, di in deeper_name_to_idx.items():
            di = int(di)
            if name in mil_name_to_idx:
                mi = int(mil_name_to_idx[name])
                if di < len(deeper_probs):
                    mil_from_deeper[mi] = deeper_probs[di]

        avg_prob = (mil_from_deeper + mil_probs) / 2.0
        pred_ens = int(np.argmax(avg_prob))
        pred_mil = int(np.argmax(mil_probs))
        pred_deeper = int(np.argmax(mil_from_deeper))

        true_name = bag_runs[0][1][j]
        true_idx = int(mil_name_to_idx[true_name])
        if pred_ens == true_idx:
            ensemble_correct += 1
        if pred_mil == true_idx:
            mil_correct += 1
        if pred_deeper == true_idx:
            deeper_correct += 1
        total += 1

    return {
        'mil_acc': mil_correct/total if total else 0.0,
        'deeper_acc': deeper_correct/total if total else 0.0,
        'ensemble_acc': ensemble_correct/total if total else 0.0,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mil', required=True)
    parser.add_argument('--deeper', required=True)
    parser.add_argument('--mil_labels', required=True)
    parser.add_argument('--deeper_labels', required=True)
    parser.add_argument('--bag_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_runs', type=int, default=1)
    parser.add_argument('--csv', default=os.path.join('Dataset','Generated_Data','wlasl_pipeline_frames.csv'))
    args = parser.parse_args()

    res = evaluate(args.mil, args.deeper, args.mil_labels, args.deeper_labels, args.csv, bag_size=args.bag_size, seed=args.seed, sample_runs=args.sample_runs)
    print('Results:', res)


if __name__ == '__main__':
    main()
