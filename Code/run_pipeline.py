"""Run the full pipeline: expand/download, convert frames, generate windows, and train.

This script uses `Code/config.py` for keywords and dataset settings. Run:
  python Code/run_pipeline.py

It will perform the steps sequentially and save the trained model under `Models/`.
"""
import os
import subprocess
import sys
sys.path.insert(0, os.getcwd())
from Code import config


def run():
  print('1) Expanding dataset (copy/download videos)')
  from Code.wlasl_expand_and_download import expand
  report = expand(config.KEYWORDS, config.TARGET_PER_CLASS, download_missing=config.DOWNLOAD_MISSING)
  print('Copied per keyword:', report)

  print('2) Converting videos to frame-level landmarks CSV')
  # call the converter
  convert_cmd = ['/opt/anaconda3/envs/ASL-Translator/bin/python', 'Code/convert_gesture_dataset.py', '--root', config.SUBSET_DIR, '--out', config.GENERATED_CSV, '--classes'] + config.KEYWORDS
  subprocess.run(convert_cmd, check=True)

  print('3) Training sequence model (TensorFlow Conv1D)')
  # run TF sequence trainer (Conv1D) with larger capacity and epochs
  train_cmd = ['/opt/anaconda3/envs/ASL-Translator/bin/python', 'Code/train_tf.py', '--csv', config.GENERATED_CSV, '--window', str(config.WINDOW_SIZE), '--stride', str(config.STRIDE), '--epochs', '30', '--batch', '32', '--out', 'gesture_wlasl_sequence']
  subprocess.run(train_cmd, check=True)

  print('Pipeline finished. Model saved under Models/gesture_wlasl_full.h5')


if __name__ == '__main__':
    run()
