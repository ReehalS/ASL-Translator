"""Scan subset videos and move unreadable/corrupt mp4s to Dataset/WLASL_corrupt/"""
import os
import shutil
import cv2

ROOT = os.path.join('Dataset', 'WLASL_subset')
CORRUPT_DIR = os.path.join('Dataset', 'WLASL_corrupt')

def is_readable(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None)
    except Exception:
        return False

def main():
    moved = 0
    checked = 0
    os.makedirs(CORRUPT_DIR, exist_ok=True)
    for root, dirs, files in os.walk(ROOT):
        for f in files:
            if not f.lower().endswith('.mp4'):
                continue
            checked += 1
            src = os.path.join(root, f)
            if not is_readable(src):
                rel = os.path.relpath(root, ROOT)
                dst_dir = os.path.join(CORRUPT_DIR, rel)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f)
                try:
                    shutil.move(src, dst)
                    print('MOVED', src, '->', dst)
                    moved += 1
                except Exception as e:
                    print('FAILED MOVE', src, e)
    print('Checked', checked, 'files; moved', moved, 'corrupt files to', CORRUPT_DIR)

if __name__ == '__main__':
    main()
