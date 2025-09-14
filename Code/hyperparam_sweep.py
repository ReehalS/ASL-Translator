"""Hyperparameter sweep for MIL models: vary learning rate, bag_size, and embed_dim.

This script trains small MIL models for a few epochs per configuration and records validation ensemble accuracy.
Results are saved to Models/hyperparam_sweep_results.json and appended to Models/gesture_results.md.
"""
import os
import json
import numpy as np
import tensorflow as tf
import itertools
import time
import sys
sys.path.insert(0, os.getcwd())
from Code.train_tf_mil import build_encoder, prepare_bags
from Code.dataset.windowed_generator import generate_windows_from_csv


def build_mil_model(window_len, feat_dim, bag_size, embed_dim, lr):
    encoder = build_encoder((window_len, feat_dim), embed_dim=embed_dim)
    if bag_size:
        bag_input = tf.keras.layers.Input(shape=(bag_size, window_len, feat_dim), dtype=tf.float32)
    else:
        bag_input = tf.keras.layers.Input(shape=(None, window_len, feat_dim), dtype=tf.float32)
    td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
    att_dense = tf.keras.layers.Dense(1)(td)
    att = tf.keras.layers.Softmax(axis=1)(att_dense)
    pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
    pooled = tf.keras.layers.Reshape((embed_dim,))(pooled)
    out = tf.keras.layers.Dense(128, activation='relu')(pooled)
    out = tf.keras.layers.Dense(1, activation='softmax')(out)  # placeholder; we'll rebuild final layer later
    # We can't set final layer num_classes until we know label map; caller will replace last layer if needed
    model = tf.keras.Model(bag_input, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    # compilation will be done by caller after adjusting final layer
    return model, encoder


def prepare_bags_local(csv, window_size=16, stride=4, bag_size=32):
    # reuse prepare_bags from train_tf_mil
    return prepare_bags(csv, window_size=window_size, stride=stride, bag_size=bag_size)


def run_sweep(csv_path, out_json='Models/hyperparam_sweep_results.json'):
    lrs = [1e-3, 5e-4, 1e-4]
    bag_sizes = [16, 32, 64]
    embeds = [64, 128, 256]
    epochs = 6
    batch = 8

    results = []
    # build fixed bag_list for each bag_size to avoid regenerating windows multiple times
    for bag_size in bag_sizes:
        bag_list, label_list = prepare_bags_local(csv_path, window_size=16, stride=4, bag_size=bag_size)
        if not bag_list:
            continue
        feat = bag_list[0].shape[2]
        # build label map
        uniq = sorted(set(label_list))
        lm = {v:i for i,v in enumerate(uniq)}
        num_classes = len(lm)

        # split indices
        N = len(bag_list)
        idx = np.arange(N)
        np.random.seed(0)
        np.random.shuffle(idx)
        split = int(0.8*N)
        train_idx, val_idx = idx[:split], idx[split:]

        for embed_dim, lr in itertools.product(embeds, lrs):
            tf.keras.backend.clear_session()
            # build encoder and model with correct output dimension
            encoder = build_encoder((16, feat), embed_dim=embed_dim)
            if bag_size:
                bag_input = tf.keras.layers.Input(shape=(bag_size, 16, feat), dtype=tf.float32)
            else:
                bag_input = tf.keras.layers.Input(shape=(None, 16, feat), dtype=tf.float32)
            td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
            att_dense = tf.keras.layers.Dense(1)(td)
            att = tf.keras.layers.Softmax(axis=1)(att_dense)
            pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
            pooled = tf.keras.layers.Reshape((embed_dim,))(pooled)
            x = tf.keras.layers.Dense(128, activation='relu')(pooled)
            out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            model = tf.keras.Model(bag_input, out)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # simple batch generator
            def batch_generator(indices, batch_size):
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i:i+batch_size]
                    max_wins = max(bag_list[j].shape[0] for j in batch_idx)
                    batch_bags = np.zeros((len(batch_idx), max_wins, 16, feat), dtype=np.float32)
                    batch_labels = np.zeros((len(batch_idx),), dtype=np.int32)
                    for ii, j in enumerate(batch_idx):
                        wins = bag_list[j]
                        batch_bags[ii, :wins.shape[0]] = wins
                        batch_labels[ii] = lm[label_list[j]]
                    yield batch_bags, batch_labels

            # train
            best_val = 0.0
            for epoch in range(epochs):
                np.random.shuffle(train_idx)
                for xb, yb in batch_generator(train_idx, batch):
                    model.train_on_batch(xb, yb)
                # val
                val_accs = []
                for xb, yb in batch_generator(val_idx, batch):
                    loss, acc = model.test_on_batch(xb, yb)
                    val_accs.append(acc)
                val_acc = float(np.mean(val_accs)) if val_accs else 0.0
            # record result
            res = dict(bag_size=bag_size, embed=embed_dim, lr=lr, val_acc=val_acc)
            print('SWEEP', res)
            results.append(res)
            # save model for top candidates (val_acc high) — keep top 3
            results_sorted = sorted(results, key=lambda r: r['val_acc'], reverse=True)
            top3 = results_sorted[:3]
            # save model file for this config if in top3
            if res in top3:
                name = f"gesture_mil_lr{lr}_bag{bag_size}_emb{embed_dim}"
                os.makedirs('Models', exist_ok=True)
                model.save(os.path.join('Models', name + '.keras'))
                with open(os.path.join('Models', name + '_labels.json'), 'w') as f:
                    json.dump(lm, f)

    # write results
    os.makedirs('Models', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    # append summary to gesture_results.md
    summary_lines = ['\n## Hyperparameter sweep results\n', '| bag_size | embed | lr | val_acc |', '|---:|---:|---:|---:|']
    for r in results:
        summary_lines.append(f"| {r['bag_size']} | {r['embed']} | {r['lr']} | {r['val_acc']:.4f} |")
    with open('Models/gesture_results.md','a') as f:
        f.write('\n'.join(summary_lines))


if __name__ == '__main__':
    csv = os.path.join('Dataset','Generated_Data','wlasl_pipeline_frames.csv')
    run_sweep(csv)
