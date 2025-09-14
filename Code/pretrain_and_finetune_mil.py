"""Pretrain per-window encoder then finetune MIL model using encoder weights.

Saves:
  - Models/encoder_pretrained.h5 (weights)
  - Models/encoder_classifier.keras (per-window classifier)
  - Models/gesture_wlasl_mil_finetuned_mil.h5 (finetuned MIL model)
  - Models/gesture_wlasl_mil_finetuned_mil_labels.json
"""
import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict
import sys
sys.path.insert(0, os.getcwd())

from Code.dataset.windowed_generator import generate_windows_from_csv
from Code.train_tf_mil import build_encoder


def build_window_dataset(csv, window=16, stride=4, jitter=0, lm_noise=0.0):
    X = []
    y = []
    for item in generate_windows_from_csv(csv, window_size=window, stride=stride, pad=True, jitter=jitter, lm_noise=lm_noise):
        X.append(item['window'])
        y.append(item['label'])
    if not X:
        return np.zeros((0, window, 0)), []
    X = np.stack(X, axis=0)
    return X, y


def train_encoder(X, y, out_prefix='encoder_pre', epochs=20, batch=64, embed_dim=128):
    # label map
    uniq = sorted(set(y))
    lm = {v:i for i,v in enumerate(uniq)}
    y_idx = np.array([lm[v] for v in y], dtype=np.int32)

    # shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]

    input_shape = X.shape[1:]
    encoder = build_encoder(input_shape, embed_dim=embed_dim)
    inp = tf.keras.layers.Input(shape=input_shape)
    x = encoder(inp)
    out = tf.keras.layers.Dense(len(uniq), activation='softmax')(x)
    clf = tf.keras.Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.summary()

    clf.fit(X[train_idx], y_idx[train_idx], validation_data=(X[val_idx], y_idx[val_idx]),
            epochs=epochs, batch_size=batch)

    # save encoder weights and classifier
    os.makedirs('Models', exist_ok=True)
    # Keras requires weight files to end with '.weights.h5'
    encoder.save_weights(os.path.join('Models', out_prefix + '.weights.h5'))
    clf.save(os.path.join('Models', out_prefix + '_classifier.keras'))
    with open(os.path.join('Models', out_prefix + '_labels.json'), 'w') as f:
        json.dump(lm, f)
    return encoder, lm


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


def finetune_mil(encoder, label_map, bag_list, label_list, out_prefix='gesture_wlasl_mil_finetuned',
                 epochs=20, batch=8, embed=128, bag_size=32, class_weight_map=None):
    # build MIL model
    num_classes = len(label_map)
    feat = bag_list[0].shape[2]
    # enforce fixed bag_size in model input to avoid variable-shaped retracing
    bag_input = tf.keras.layers.Input(shape=(bag_size, bag_list[0].shape[1], feat), dtype=tf.float32)
    td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
    att_dense = tf.keras.layers.Dense(1)(td)
    att = tf.keras.layers.Softmax(axis=1)(att_dense)
    pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
    pooled = tf.keras.layers.Reshape((embed,))(pooled)
    out = tf.keras.layers.Dense(128, activation='relu')(pooled)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(out)
    model = tf.keras.Model(bag_input, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # map labels to indices
    lm = label_map
    y = np.array([lm[l] for l in label_list])

    N = len(bag_list)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]

    def batch_generator(indices, batch_size):
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            # use fixed bag_size (prepared earlier) for stable shapes
            batch_bags = np.zeros((len(batch_idx), bag_size, bag_list[0].shape[1], feat), dtype=np.float32)
            batch_labels = np.zeros((len(batch_idx),), dtype=np.int32)
            batch_weights = np.ones((len(batch_idx),), dtype=np.float32)
            for ii, j in enumerate(batch_idx):
                wins = bag_list[j]
                # wins should already be bag_size x L x feat
                batch_bags[ii] = wins
                batch_labels[ii] = lm[label_list[j]]
                if class_weight_map is not None:
                    batch_weights[ii] = class_weight_map.get(batch_labels[ii], 1.0)
            yield batch_bags, batch_labels, batch_weights

    best_val = 0.0
    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        for xb, yb, w in batch_generator(train_idx, batch):
            model.train_on_batch(xb, yb, sample_weight=w)
        # val
        val_accs = []
        for xb, yb, w in batch_generator(val_idx, batch):
            loss, acc = model.test_on_batch(xb, yb, sample_weight=w)
            val_accs.append(acc)
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        print(f'Epoch {epoch+1}/{epochs} val_acc={val_acc:.3f}')
        if val_acc > best_val:
            best_val = val_acc
            # save both native keras and weights for stable reloads
            os.makedirs('Models', exist_ok=True)
            model.save(os.path.join('Models', out_prefix + '_mil.keras'))
            model.save_weights(os.path.join('Models', out_prefix + '_mil.weights.h5'))

    with open(os.path.join('Models', out_prefix + '_mil_labels.json'), 'w') as f:
        json.dump(lm, f)
    print('Best val acc', best_val)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jitter', type=int, default=0)
    parser.add_argument('--lm_noise', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--oversample', action='store_true', help='Oversample low-count classes in window dataset')
    parser.add_argument('--bag_size', type=int, default=32, help='Fixed bag size used for MIL training/eval')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights during MIL finetune')
    args = parser.parse_args()

    csv = os.path.join('Dataset', 'Generated_Data', 'wlasl_pipeline_frames.csv')
    X, y = build_window_dataset(csv, window=16, stride=4, jitter=args.jitter, lm_noise=args.lm_noise)
    print('Loaded windows', X.shape)
    # optionally oversample: simple per-class upsampling to match max class count
    if args.oversample and len(y) > 0:
        from collections import defaultdict
        class_to_idxs = defaultdict(list)
        for i, lab in enumerate(y):
            class_to_idxs[lab].append(i)
        maxc = max(len(v) for v in class_to_idxs.values())
        new_X = [X]
        new_y = [y]
        X_list = [X]
        y_list = list(y)
        # build new arrays by sampling with replacement
        X_resampled = []
        y_resampled = []
        for lab, idxs in class_to_idxs.items():
            cur = X[idxs]
            need = maxc - cur.shape[0]
            X_resampled.append(cur)
            y_resampled.extend([lab] * cur.shape[0])
            if need > 0:
                choices = np.random.choice(len(idxs), need, replace=True)
                X_resampled.append(cur[choices])
                y_resampled.extend([lab] * need)
        X = np.concatenate(X_resampled, axis=0)
        y = np.array(y_resampled)
        print('After oversample windows', X.shape)

    encoder, label_map = train_encoder(X, y, out_prefix='encoder_pre', epochs=args.epochs, batch=args.batch)

    # prepare bags for finetune
    bag_list, label_list = prepare_bags(csv, window_size=16, stride=4, bag_size=args.bag_size)
    # ensure label_map matches bag labels; if not, rebuild mapping
    uniq = sorted(set(label_list))
    lm = {v:i for i,v in enumerate(uniq)}
    # remap encoder label map not necessary for MIL; we use LM from bag labels

    # load encoder weights into a fresh encoder instance
    feat = bag_list[0].shape[2]
    enc = build_encoder((16, feat), embed_dim=128)
    enc.load_weights(os.path.join('Models', 'encoder_pre.weights.h5'))

    # optional: compute class weight map for MIL
    class_weight_map = None
    if args.use_class_weights:
        from collections import Counter
        counts = Counter(label_list)
        maxc = max(counts.values())
        class_weight_map = {i: float(maxc / counts[label]) for i, label in enumerate(sorted(counts.keys()))}

    finetune_mil(enc, lm, bag_list, label_list, out_prefix='gesture_wlasl_mil_finetuned',
                 epochs=args.epochs, batch=8, embed=128, bag_size=args.bag_size, class_weight_map=class_weight_map)


if __name__ == '__main__':
    main()
