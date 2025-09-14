"""Deeper Conv1D + residual blocks and optional BiLSTM head trainer."""
import os
import numpy as np
from collections import defaultdict
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf

from Code.dataset.windowed_generator import windows_to_numpy


def label_map(labels):
    uniq = sorted(set(labels))
    lm = {v: i for i, v in enumerate(uniq)}
    return lm


def residual_block(x, filters, kernel_size=3, stride=1, dropout=0.2):
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    conv = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(conv)
    if x.shape[-1] != filters:
        x = tf.keras.layers.Conv1D(filters, 1, padding='same')(x)
    out = tf.keras.layers.Add()([x, conv])
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    return out


def build_deeper_model(input_shape, num_classes, use_bilstm=False):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(inp)
    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = residual_block(x, 256)
    x = tf.keras.layers.MaxPool1D(2)(x)

    x = tf.keras.layers.Conv1D(512, 3, padding='same', activation='relu')(x)
    x = residual_block(x, 512)

    if use_bilstm:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False))(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    X, y, clips = windows_to_numpy(args.csv, window_size=args.window, stride=args.stride, jitter=args.jitter, lm_noise=args.lm_noise)
    if X.size == 0:
        print('No windows found')
        return
    lm = label_map(y)
    y_idx = np.array([lm[v] for v in y], dtype=np.int32)

    # optional oversample at window-level to balance classes
    if args.oversample:
        from collections import defaultdict
        cls_to_idx = defaultdict(list)
        for i, lab in enumerate(y):
            cls_to_idx[lab].append(i)
        maxc = max(len(v) for v in cls_to_idx.values())
        X_parts = []
        y_parts = []
        for lab, idxs in cls_to_idx.items():
            cur = X[idxs]
            need = maxc - cur.shape[0]
            X_parts.append(cur)
            y_parts.extend([lab] * cur.shape[0])
            if need > 0:
                choices = np.random.choice(len(idxs), need, replace=True)
                X_parts.append(cur[choices])
                y_parts.extend([lab] * need)
        X = np.concatenate(X_parts, axis=0)
        y = np.array(y_parts)
        y_idx = np.array([lm[v] for v in y], dtype=np.int32)

    # clip-wise split
    clip_to_idx = defaultdict(list)
    for i, c in enumerate(clips):
        clip_to_idx[c].append(i)
    clip_ids = list(clip_to_idx.keys())
    split = int(0.8 * len(clip_ids))
    train_clips = set(clip_ids[:split])
    train_idx = [i for i, c in enumerate(clips) if c in train_clips]
    val_idx = [i for i, c in enumerate(clips) if c not in train_clips]

    X_train = X[train_idx]
    y_train = y_idx[train_idx]
    X_val = X[val_idx]
    y_val = y_idx[val_idx]

    N, L, D = X_train.shape
    model = build_deeper_model((L, D), num_classes=len(lm), use_bilstm=args.bilstm)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch, callbacks=callbacks)

    os.makedirs('Models', exist_ok=True)
    out_path = os.path.join('Models', args.out + '.keras')
    model.save(out_path)
    import json
    with open(os.path.join('Models', args.out + '_labels.json'), 'w') as f:
        json.dump(lm, f)
    print('Saved model to', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--jitter', type=int, default=0)
    parser.add_argument('--lm_noise', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', type=str, default='gesture_wlasl_deeper')
    parser.add_argument('--bilstm', action='store_true')
    parser.add_argument('--oversample', action='store_true', help='Oversample windows per class')
    args = parser.parse_args()
    train(args)
