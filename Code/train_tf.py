"""Train a lightweight Conv1D temporal model using TensorFlow/Keras on windowed data."""
import os
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.getcwd())

import tensorflow as tf

from Code.dataset.windowed_generator import windows_to_numpy


def label_map(labels):
    uniq = sorted(set(labels))
    lm = {v: i for i, v in enumerate(uniq)}
    return lm


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    X, y, clips = windows_to_numpy(args.csv, window_size=args.window, stride=args.stride)
    if X.size == 0:
        print('No windows found')
        return
    lm = label_map(y)
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
    model = build_model((L, D), num_classes=len(lm))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch, callbacks=callbacks)

    os.makedirs('Models', exist_ok=True)
    out_path = os.path.join('Models', args.out + '.keras')
    # save in native Keras format to avoid legacy HDF5 warnings
    model.save(out_path)
    # save label map
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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', type=str, default='gesture_wlasl_tf')
    args = parser.parse_args()
    train(args)
