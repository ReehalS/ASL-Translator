"""Helper for preparing WLASL (Kaggle) subset for training.

Modes:
 - download: use Kaggle API to download a specified Kaggle dataset (you must have `kaggle` CLI configured)
 - prepare: given a local folder with videos (e.g., extracted from kaggle), restructure into class folders and optionally call convert_gesture_dataset.py

Usage examples:
  # If you have kaggle CLI configured and a WLASL dataset entry on Kaggle:
  python Code/wlasl_kaggle_prepare.py --mode download --kaggle-dataset <owner/dataset-name> --out-dir data/wlasl_raw

  # If you already downloaded and extracted Kaggle dataset locally:
  python Code/wlasl_kaggle_prepare.py --mode prepare --input-dir data/wlasl_raw --out-dir data/wlasl_clips

Notes:
- This script does not attempt to download from YouTube; it assumes Kaggle dataset contains ready-to-use clips or provide a metadata CSV to allow extraction.
- For small experiments (few symbols), place a small subset of clips in folders named by label and run `Code/convert_gesture_dataset.py`.
"""
import argparse
import os
import shutil
from pathlib import Path
import subprocess


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def download_kaggle(dataset, out_dir):
    # Requires kaggle CLI configured (kaggle datasets download owner/dataset -p out_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"]) 


def prepare_local(input_dir, out_dir):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Heuristic: if input_dir contains class-labeled subfolders, just copy/symlink them
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            # copy folder structure
            dst = out_dir / child.name
            if not dst.exists():
                shutil.copytree(child, dst)
        else:
            # if flat list of videos with names like <label>__<clipid>.mp4, try to split
            name = child.name
            if '__' in name:
                label = name.split('__')[0]
                dst = out_dir / label
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, dst / child.name)
    print(f"Prepared clips in {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["download", "prepare"], required=True)
    p.add_argument("--kaggle-dataset", help="owner/dataset-name for kaggle datasets download")
    p.add_argument("--input-dir")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if args.mode == "download":
        if not args.kaggle_dataset:
            raise SystemExit("--kaggle-dataset required for download mode")
        download_kaggle(args.kaggle_dataset, args.out_dir)
    else:
        if not args.input_dir:
            raise SystemExit("--input-dir required for prepare mode")
        prepare_local(args.input_dir, args.out_dir)

if __name__ == "__main__":
    main()
