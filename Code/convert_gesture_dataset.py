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


def extract_landmarks_from_image(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        lm = results.multi_hand_landmarks[0]
        data = []
        for p in lm.landmark:
            data.append(p.x)
            data.append(p.y)
        if len(data) == 42:
            return data
    return None


def process_video(path, hands, writer, label=None, step=1):
    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            data = extract_landmarks_from_image(frame, hands)
            if data is not None:
                row = data + [label]
                writer.writerow(row)
        frame_idx += 1
    cap.release()


def process_image_dir(path, hands, writer, label=None):
    images = sorted(glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png")))
    for imgp in images:
        img = cv2.imread(imgp)
        if img is None:
            continue
        data = extract_landmarks_from_image(img, hands)
        if data is not None:
            writer.writerow(data + [label])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--video-ext", default="mp4")
    p.add_argument("--step", type=int, default=2, help="Process every Nth frame from videos")
    args = p.parse_args()

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        # header: x0,y0,...,x20,y20,clip_id,label
        header = []
        for i in range(21):
            header.append(f"x{i}")
            header.append(f"y{i}")
        header += ["clip_id", "label"]
        writer.writerow(header)

        # walk input dir
        if args.recursive:
            for root, dirs, files in os.walk(args.input_dir):
                # if folder contains videos/images, treat folder name as label
                # heuristic: if contains video files, process each video as that label
                vids = glob.glob(os.path.join(root, f"*.{args.video_ext}"))
                if vids:
                    label = os.path.basename(root)
                    for v in vids:
                        clip_id = os.path.splitext(os.path.basename(v))[0]
                        process_video(v, hands, writer, label=label, step=args.step)
                # image folders
                imgs = glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png"))
                if imgs:
                    label = os.path.basename(root)
                    process_image_dir(root, hands, writer, label=label)
        else:
            # non-recursive: input-dir may contain subfolders per class
            for child in sorted(Path(args.input_dir).iterdir()):
                if child.is_dir():
                    label = child.name
                    # check for videos first
                    vids = list(child.glob(f"*.{args.video_ext}"))
                    if vids:
                        for v in vids:
                            clip_id = os.path.splitext(os.path.basename(v))[0]
                            process_video(v, hands, writer, label=label, step=args.step)
                    else:
                        process_image_dir(child, hands, writer, label=label)

    hands.close()


if __name__ == "__main__":
    main()
