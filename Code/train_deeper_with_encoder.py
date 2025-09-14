"""Train deeper model but initialize lower conv layers with encoder weights when available.

This is a lightweight adapter: it will load the pretrained encoder weights from
Models/encoder_pre.weights.h5 (if present), then train the deeper model for a small number
of epochs to validate transfer benefit.
"""
import os
import numpy as np
import json
import tensorflow as tf
import sys
sys.path.insert(0, os.getcwd())
from Code.dataset.windowed_generator import windows_to_numpy
from Code.train_tf_deeper import build_deeper_model
from Code.train_tf_mil import build_encoder


def transfer_train(csv, out='gesture_wlasl_deeper_encinit', epochs=5, batch=32, window=16, stride=4, jitter=0, lm_noise=0.0, bilstm=False):
    X, y, clips = windows_to_numpy(csv, window_size=window, stride=stride, jitter=jitter, lm_noise=lm_noise)
    if X.size == 0:
        print('No windows')
        return
    uniq = sorted(set(y))
    lm = {v:i for i,v in enumerate(uniq)}
    y_idx = np.array([lm[v] for v in y], dtype=np.int32)

    # clip split
    from collections import defaultdict
    clip_to_idx = defaultdict(list)
    for i, c in enumerate(clips):
        clip_to_idx[c].append(i)
    clip_ids = list(clip_to_idx.keys())
    split = int(0.8 * len(clip_ids))
    train_clips = set(clip_ids[:split])
    train_idx = [i for i,c in enumerate(clips) if c in train_clips]
    val_idx = [i for i,c in enumerate(clips) if c not in train_clips]

    X_train = X[train_idx]
    y_train = y_idx[train_idx]
    X_val = X[val_idx]
    y_val = y_idx[val_idx]

    model = build_deeper_model((X.shape[1], X.shape[2]), num_classes=len(lm), use_bilstm=bilstm)

    # attempt to load encoder weights and copy conv weights into first conv layers if shapes match
    enc_weights_path = os.path.join('Models','encoder_pre.weights.h5')
    if os.path.exists(enc_weights_path):
        try:
            enc = build_encoder((window, X.shape[2]), embed_dim=128)
            enc.load_weights(enc_weights_path)
            # find first Conv1D in encoder and first conv in deeper and copy weights where possible
            enc_conv = [l for l in enc.layers if isinstance(l, tf.keras.layers.Conv1D)]
            deep_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv1D)]
            for e,d in zip(enc_conv, deep_conv):
                ew = e.get_weights()
                dw = d.get_weights()
                if len(ew) and len(dw) and ew[0].shape == dw[0].shape:
                    d.set_weights(ew)
            print('Transferred encoder conv weights into deeper model')
        except Exception as e:
            print('Encoder transfer failed:', e)

    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch)
    os.makedirs('Models', exist_ok=True)
    out_path = os.path.join('Models', out + ('.bilstm.keras' if bilstm else '.keras'))
    model.save(out_path)
    with open(os.path.join('Models', out + '_labels.json'), 'w') as f:
        json.dump(lm, f)
    print('Saved', out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join('Dataset','Generated_Data','wlasl_pipeline_frames.csv'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='gesture_wlasl_deeper_encinit')
    parser.add_argument('--bilstm', action='store_true')
    args = parser.parse_args()
    transfer_train(args.csv, out=args.out, epochs=args.epochs, batch=args.batch, bilstm=args.bilstm)
