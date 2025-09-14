"""
Realtime ASL demo script (mirrors notebooks/gesture_realtime.ipynb)
Usage:
  python Code/realtime_demo.py --dry-run   # test model loading and weights
  python Code/realtime_demo.py            # run webcam loop (requires camera and display)

"""
import argparse
import os
import sys
import time
import json
from collections import deque

import numpy as np
import tensorflow as tf

# optional imports for runtime
try:
    import cv2
    import mediapipe as mp
except Exception:
    cv2 = None
    mp = None

# allow running from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# paths (adjust if needed)
MIL_MODEL = os.path.join(ROOT, 'Models', 'gesture_wlasl_mil_finetuned_mil.keras')
DEEPER_MODEL = os.path.join(ROOT, 'Models', 'gesture_wlasl_deeper_encinit_long.keras')
DEEPER_MODEL_2 = os.path.join(ROOT, 'Models', 'gesture_wlasl_deeper_encinit_long.bilstm.keras')
MIL_LABELS = os.path.join(ROOT, 'Models', 'gesture_wlasl_mil_finetuned_mil_labels.json')
DEEPER_LABELS = os.path.join(ROOT, 'Models', 'gesture_wlasl_deeper_encinit_long_labels.json')
DEEPER2_LABELS = DEEPER_LABELS


def load_labels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        return json.load(f)


def load_model_try(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        print('Full model load failed for', path, '->', e)
        return None


def build_mil_with_fallback(win, feat, bag_size, mil_labels, mil_model_path):
    # Build the MIL architecture (encoder -> TimeDistributed -> attention -> pool -> dense)
    # We always construct the architecture here so we can expose attention weights for visualization.
    print('Building MIL architecture (for inference + attention visualization)...')
    # try package import first
    try:
        from Code.train_tf_mil import build_encoder
        print('Imported build_encoder from Code.train_tf_mil')
    except Exception:
        # file-based import with repo root on sys.path
        import importlib.util
        candidates = [
            os.path.join(ROOT, 'Code', 'train_tf_mil.py'),
            os.path.join(os.path.dirname(ROOT), 'Code', 'train_tf_mil.py'),
        ]
        found = None
        for p in candidates:
            if os.path.exists(p):
                found = os.path.abspath(p)
                break
        if found is None:
            raise RuntimeError('Cannot find Code/train_tf_mil.py; looked in: %s' % (candidates,))
        repo_root = os.path.abspath(os.path.join(found, '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        spec = importlib.util.spec_from_file_location('train_tf_mil_notebook_helper', found)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, 'build_encoder'):
            raise RuntimeError('Loaded module from %s but it does not export build_encoder' % found)
        build_encoder = mod.build_encoder
        print('Loaded build_encoder from', found)
    # build encoder and MIL (and attention model)
    encoder = build_encoder((win, feat), embed_dim=128)
    bag_input = tf.keras.layers.Input(shape=(bag_size, win, feat), dtype=tf.float32)
    td = tf.keras.layers.TimeDistributed(encoder)(bag_input)
    att_dense = tf.keras.layers.Dense(1)(td)
    att = tf.keras.layers.Softmax(axis=1, name='mil_attention')(att_dense)
    pooled = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True), name='pool')([td, att])
    pooled = tf.keras.layers.Reshape((128,))(pooled)
    out = tf.keras.layers.Dense(128, activation='relu')(pooled)
    out = tf.keras.layers.Dense(len(mil_labels), activation='softmax')(out)
    mil = tf.keras.Model(bag_input, out)
    # attention model outputs the per-window attention weights (shape (batch, bag_size, 1))
    att_model = tf.keras.Model(bag_input, att)
    # try weights
    weights_loaded = False
    try:
        if os.path.exists(mil_model_path + '.weights.h5'):
            mil.load_weights(mil_model_path + '.weights.h5')
            weights_loaded = True
            print('Loaded MIL weights from', mil_model_path + '.weights.h5')
    except Exception as e:
        print('Failed to load .weights.h5:', e)
    if not weights_loaded:
        try:
            mil.load_weights(mil_model_path)
            weights_loaded = True
            print('Loaded MIL weights from', mil_model_path)
        except Exception as e:
            print('Failed to load MIL weights from', mil_model_path, '->', e)
    if not weights_loaded:
        print('Warning: MIL weights not found; MIL model will be uninitialized and predictions will be meaningless')
    return mil, att_model


def main(dry_run=False):
    mil_labels = load_labels(MIL_LABELS)
    deeper_labels = load_labels(DEEPER_LABELS)
    deeper2_labels = load_labels(DEEPER2_LABELS)

    deeper = load_model_try(DEEPER_MODEL)
    deeper2 = None
    if os.path.exists(DEEPER_MODEL_2):
        deeper2 = load_model_try(DEEPER_MODEL_2)

    # infer window/feat from deeper if available
    default_window = 16
    default_feat = 126
    win, feat = default_window, default_feat
    if deeper is not None:
        try:
            shape = getattr(deeper, 'input_shape', None)
            if shape is None:
                win, feat = default_window, default_feat
            else:
                if len(shape) >= 3:
                    win = int(shape[1]); feat = int(shape[2])
                elif len(shape) == 2:
                    win = int(shape[0]); feat = int(shape[1])
        except Exception:
            win, feat = default_window, default_feat

    bag_size = 32
    mil, att_model = build_mil_with_fallback(win, feat, bag_size, mil_labels, MIL_MODEL)

    print('Using window_len=', win, 'feat=', feat, 'bag_size=', bag_size)

    if dry_run:
        print('Dry-run complete: models loaded (or rebuilt). Exiting.')
        return

    if cv2 is None or mp is None:
        raise RuntimeError('cv2 or mediapipe not available; cannot run webcam loop')

    # setup mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit('Cannot open webcam')

    frame_buffer = deque(maxlen=win)
    bag_buffer = deque(maxlen=bag_size)  # stores recent windows for MIL bag
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # extract landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            feats = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                vals = []
                for p in lm.landmark:
                    vals.extend([p.x, p.y, p.z])
                feats = np.array(vals, dtype=np.float32)
            if feats is not None:
                frame_buffer.append(feats)
            # compute and draw hand bbox + landmarks for visualization
            if res.multi_hand_landmarks:
                # compute bbox in pixel coords from normalized landmarks
                lm = res.multi_hand_landmarks[0]
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                h, w, _ = frame.shape
                x_min = int(max(0, min(xs) * w) - 5)
                x_max = int(min(w - 1, max(xs) * w) + 5)
                y_min = int(max(0, min(ys) * h) - 5)
                y_max = int(min(h - 1, max(ys) * h) + 5)
                # translucent overlay for bbox
                try:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0,128,255), -1)
                    alpha = 0.15
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                except Exception:
                    pass
                # draw bbox border and landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,128,255), 2)
                for p in lm.landmark:
                    px = int(p.x * w); py = int(p.y * h)
                    cv2.circle(frame, (px, py), 2, (255,255,255), -1)
            label_text = 'No hand'
            if len(frame_buffer) >= win:
                arr = np.array(list(frame_buffer)[-win:], dtype=np.float32)
                if arr.shape[1] < feat:
                    pad = np.zeros((arr.shape[0], feat - arr.shape[1]), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=1)
                elif arr.shape[1] > feat:
                    arr = arr[:, :feat]
                window = arr
                deeper_probs = deeper.predict(window[np.newaxis,...], verbose=0)[0] if deeper is not None else np.zeros((len(mil_labels),))
                mil_from_deeper = np.zeros(len(mil_labels), dtype=float)
                for name, di in deeper_labels.items():
                    di = int(di)
                    if name in mil_labels and di < len(deeper_probs):
                        mil_from_deeper[int(mil_labels[name])] = deeper_probs[di]
                # update bag buffer with the latest window
                bag_buffer.append(window)
                # if bag not full, pad by repeating newest window
                if len(bag_buffer) < bag_size:
                    pads = [window] * (bag_size - len(bag_buffer))
                    bag = np.stack(pads + list(bag_buffer), axis=0)
                else:
                    bag = np.stack(list(bag_buffer), axis=0)
                mil_input = np.expand_dims(bag, axis=0)
                mil_probs = mil.predict(mil_input, verbose=0)[0]
                # compute attention weights for visualization
                try:
                    att_w = att_model.predict(mil_input, verbose=0)[0].squeeze()
                except Exception:
                    att_w = np.ones((bag.shape[0],), dtype=float) / float(bag.shape[0])
                avg_prob = (mil_from_deeper + mil_probs) / 2.0
                pred_idx = int(np.argmax(avg_prob))
                pred_name = None
                for k,v in mil_labels.items():
                    if int(v) == pred_idx:
                        pred_name = k; break
                label_text = pred_name if pred_name is not None else 'unknown'
            # overlay prediction text
            cv2.putText(frame, f'Pred: {label_text}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # draw top-3 probabilities as a small bar chart
            try:
                topk = 3
                top_idx = np.argsort(avg_prob)[-topk:][::-1]
                h0 = 60
                for i, idx in enumerate(top_idx):
                    name = None
                    for k, v in mil_labels.items():
                        if int(v) == int(idx):
                            name = k; break
                    prob = float(avg_prob[int(idx)])
                    text = f'{name}:{prob:.2f}' if name is not None else f'{int(idx)}:{prob:.2f}'
                    cv2.putText(frame, text, (10, h0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            except Exception:
                pass
            # draw attention bars on right side (orange bars show per-window attention weights
            # produced by the MIL model; taller bars = model paid more attention to that window)
            # These help visualize which recent windows the MIL attention mechanism finds important.
            # draw attention bars on right side
            try:
                h, w, _ = frame.shape
                max_w = 120
                start_x = w - max_w - 10
                start_y = 50
                for i, aw in enumerate(att_w[-20:][::-1]):
                    # show only up to last 20 windows to avoid clutter
                    by = start_y + i*8
                    bw = int(aw * max_w)
                    cv2.rectangle(frame, (start_x, by), (start_x + bw, by + 6), (0,200,255), -1)
            except Exception:
                pass
            cv2.imshow('ASL Realtime', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', help='Load models and exit (no camera)')
    args = p.parse_args()
    main(dry_run=args.dry_run)
