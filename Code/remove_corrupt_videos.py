"""Scan roots for unreadable/corrupt mp4s and move them to Dataset/WLASL_corrupt/.

Usage:
    python remove_corrupt_videos.py [root1 root2 ...]

If no roots are provided the script will scan:
    Dataset/WLASL_subset and Dataset/WLASL

Files are moved to Dataset/WLASL_corrupt/<relative-path>/file.mp4
"""
import os
import shutil
import sys
import cv2

DEFAULT_ROOTS = [os.path.join('Dataset', 'WLASL_subset'), os.path.join('Dataset', 'WLASL')]
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


def scan_and_move(root):
    moved = 0
    checked = 0
    if not os.path.exists(root):
        return checked, moved
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith('.mp4'):
                continue
            checked += 1
            src = os.path.join(dirpath, f)
            if not is_readable(src):
                rel = os.path.relpath(dirpath, root)
                dst_dir = os.path.join(CORRUPT_DIR, rel)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, f)
                try:
                    shutil.move(src, dst)
                    print('MOVED', src, '->', dst)
                    moved += 1
                except Exception as e:
                    print('FAILED MOVE', src, e)
    return checked, moved


def main(argv=None):
    argv = argv or sys.argv[1:]
    roots = argv if argv else DEFAULT_ROOTS
    os.makedirs(CORRUPT_DIR, exist_ok=True)
    total_checked = 0
    total_moved = 0
    for r in roots:
        checked, moved = scan_and_move(r)
        print(f'Scanned {r}: checked {checked} files, moved {moved} corrupt')
        total_checked += checked
        total_moved += moved
    print('TOTAL: Checked', total_checked, 'files; moved', total_moved, 'corrupt files to', CORRUPT_DIR)


if __name__ == '__main__':
    main()
