"""Train using Multiple Instance Learning (MIL) with attention pooling on windowed landmark data.

Each clip is a bag of windows; the model computes window embeddings and attention weights, pools to a clip representation, and predicts the clip label.
"""
import os
import json
import numpy as np
from collections import defaultdict
import tensorflow as tf
import sys
sys.path.insert(0, os.getcwd())

from Code.dataset.windowed_generator import generate_windows_from_csv


def build_encoder(window_shape, embed_dim=128):
    inp = tf.keras.layers.Input(shape=window_shape)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(embed_dim, activation='relu')(x)
    return tf.keras.Model(inp, x, name='encoder')


def attention_pooling(embeds):
    # embeds: (num_windows, embed_dim)
    w = tf.keras.layers.Dense(1)(embeds)  # (num_windows, 1)
    w = tf.keras.layers.Softmax(axis=0)(w)
    pooled = tf.reduce_sum(w * embeds, axis=0)
    return pooled, w


def prepare_bags(csv_path, window_size=16, stride=4, bag_size=None):
    # create dict clip_id -> list of window arrays
    bags = defaultdict(list)
    labels = {}
    for item in generate_windows_from_csv(csv_path, window_size=window_size, stride=stride, pad=True):
        bags[item['clip_id']].append(item['window'])
        labels[item['clip_id']] = item['label']
    # convert to lists
    bag_list = []
    label_list = []
    for clip_id, wins in bags.items():
        arr = np.stack(wins, axis=0)
        if bag_size is not None:
            # if more windows than bag_size, sample without replacement
            if arr.shape[0] >= bag_size:
                idxs = np.random.choice(arr.shape[0], bag_size, replace=False)
                arr = arr[idxs]
            else:
                # pad by repeating last window until bag_size
                repeats = bag_size - arr.shape[0]
                pad = np.repeat(arr[-1:], repeats, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
        bag_list.append(arr)
        label_list.append(labels[clip_id])
    return bag_list, label_list


def label_map(labels):
    uniq = sorted(set(labels))
    return {v:i for i,v in enumerate(uniq)}


def train(args):
    bag_list, label_list = prepare_bags(args.csv, window_size=args.window, stride=args.stride, bag_size=args.bag_size)
    if not bag_list:
        print('No bags found')
        return
    lm = label_map(label_list)
    y = np.array([lm[l] for l in label_list])

    # split train/val
    N = len(bag_list)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    encoder = build_encoder((args.window, bag_list[0].shape[2]), embed_dim=args.embed)
    num_classes = len(lm)

    # build model for a single bag by creating inputs of shape (bag_size, window, feat) if bag_size provided
    if args.bag_size:
        bag_input = tf.keras.layers.Input(shape=(args.bag_size, args.window, bag_list[0].shape[2]), dtype=tf.float32)
    else:
        bag_input = tf.keras.layers.Input(shape=(None, args.window, bag_list[0].shape[2]), dtype=tf.float32)
    # time-distributed encoder
    td = tf.keras.layers.TimeDistributed(encoder)(bag_input)  # (batch, num_windows, embed_dim)
    # apply attention along num_windows axis
    att_dense = tf.keras.layers.Dense(1)(td)
    att = tf.keras.layers.Softmax(axis=1)(att_dense)
    pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
    pooled = tf.keras.layers.Reshape((args.embed,))(pooled)
    out = tf.keras.layers.Dense(128, activation='relu')(pooled)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(out)
    model = tf.keras.Model(bag_input, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # training loop: feed bags in batches (pad num_windows to max in batch)
    def batch_generator(indices, batch_size):
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            max_wins = max(bag_list[j].shape[0] for j in batch_idx)
            batch_bags = np.zeros((len(batch_idx), max_wins, args.window, bag_list[0].shape[2]), dtype=np.float32)
            batch_labels = np.zeros((len(batch_idx),), dtype=np.int32)
            for ii, j in enumerate(batch_idx):
                wins = bag_list[j]
                batch_bags[ii, :wins.shape[0]] = wins
                batch_labels[ii] = lm[label_list[j]]
            yield batch_bags, batch_labels

    best_val = 0.0
    for epoch in range(args.epochs):
        np.random.shuffle(train_idx)
        for xb, yb in batch_generator(train_idx, args.batch):
            model.train_on_batch(xb, yb)
        # val
        val_losses = []
        val_accs = []
        for xb, yb in batch_generator(val_idx, args.batch):
            loss, acc = model.test_on_batch(xb, yb)
            val_losses.append(loss)
            val_accs.append(acc)
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        print(f'Epoch {epoch+1}/{args.epochs} val_acc={val_acc:.3f}')
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs('Models', exist_ok=True)
            model.save(os.path.join('Models', args.out + '_mil.keras'))
            model.save_weights(os.path.join('Models', args.out + '_mil.weights.h5'))

    # save label map
    with open(os.path.join('Models', args.out + '_mil_labels.json'), 'w') as f:
        json.dump(lm, f)
    print('Best val acc', best_val)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--embed', type=int, default=128)
    parser.add_argument('--bag_size', type=int, default=None,
                        help='If set, sample/pad each clip to this many windows')
    parser.add_argument('--out', type=str, default='gesture_wlasl_mil')
    args = parser.parse_args()
    train(args)
