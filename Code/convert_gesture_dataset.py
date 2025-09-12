"""Conversion helper: turn videos or image-sequence gesture datasets into per-frame MediaPipe landmarks CSV compatible with this project.

Usage (examples):
  python Code/convert_gesture_dataset.py --input-dir /path/to/dataset --output csvs/output.csv --label-labels-file labels.txt

This script processes video files or directories of images, runs MediaPipe Hands, and writes rows in the same format as existing CSVs: x0,y0,...,x40,y40,label

It keeps a small CLI and supports recursion. It's a template; adjust paths and any dataset-specific mapping as needed.
"""
import os
import argparse
import glob
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
import csv
import os
import mediapipe as mp
import cv2
import time

mp_hands = mp.solutions.hands


def extract_landmarks_from_image(image, hands):
    """Return dict with left and right flattened landmarks or None if none detected.
    Output format: {'left': [x0,y0,z0,...], 'right': [x0,y0,z0,...], 'timestamp': float}
    """
    h, w = image.shape[:2]
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out = {'left': None, 'right': None}
    if not results.multi_hand_landmarks:
        return out
    # mediapipe gives a parallel list of handedness; map them
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'Left' or 'Right'
            flattened = []
            for lm in hand_landmarks.landmark:
                # store normalized coords and z
                flattened.extend([lm.x, lm.y, lm.z])
            if label.startswith('l'):
                out['left'] = flattened
            else:
                out['right'] = flattened
    return out


def process_video(path, hands, writer, label=None, step=1):
    cap = cv2.VideoCapture(path)
    # quick readability check: try to read a single frame
    if not cap.isOpened():
        print('SKIP unreadable video (cannot open):', path)
        return
    ret, _ = cap.read()
    if not ret:
        cap.release()
        print('SKIP unreadable video (no frames):', path)
        return
    frame_idx = 0
    clip_id = os.path.splitext(os.path.basename(path))[0]
    start_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            lm = extract_landmarks_from_image(frame, hands)
            timestamp = frame_idx / fps
            # build row: clip_id,label,frame_idx,timestamp, left(63), right(63)
            left = lm.get('left') or [None] * (21 * 3)
            right = lm.get('right') or [None] * (21 * 3)
            row = [clip_id, label, frame_idx, timestamp] + left + right
            writer.writerow(row)
        frame_idx += 1
    cap.release()


def process_image_dir(path, hands, writer, label=None):
    files = sorted(os.listdir(path))
    clip_id = os.path.basename(path)
    for i, f in enumerate(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
        lm = extract_landmarks_from_image(img, hands)
        timestamp = i
        left = lm.get('left') or [None] * (21 * 3)
        right = lm.get('right') or [None] * (21 * 3)
        row = [clip_id, label, i, timestamp] + left + right
        writer.writerow(row)


def walk_and_process(root, out_csv, classes=None):
    # header: clip_id,label,frame_idx,timestamp, lx0,ly0,lz0,..., rx0,ry0,rz0,...
    header = ['clip_id', 'label', 'frame_idx', 'timestamp']
    for side in ('l', 'r'):
        for i in range(21):
            header += [f'{side}x{i}', f'{side}y{i}', f'{side}z{i}']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
            for dirpath, dirnames, filenames in os.walk(root):
                # process directories that contain images (per-clip image folders)
                for d in sorted(dirnames):
                    label = d
                    if classes and label not in classes:
                        continue
                    full = os.path.join(dirpath, d)
                    if os.path.isdir(full) and any(fn.lower().endswith(('.png', '.jpg', '.jpeg')) for fn in os.listdir(full)):
                        process_image_dir(full, hands, writer, label=label)
                # process video files in this directory
                for fn in sorted(filenames):
                    if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        label = os.path.basename(dirpath)
                        if classes and label not in classes:
                            continue
                        process_video(os.path.join(dirpath, fn), hands, writer, label=label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--classes', nargs='*')
    args = parser.parse_args()
    walk_and_process(args.root, args.out, classes=args.classes)
