"""Windowed dataset generator for frame-level MediaPipe CSVs.

Produces sliding windows of shape (L, D) per clip for training temporal models.
"""
from typing import List, Tuple, Iterator, Dict
import numpy as np
import pandas as pd


def load_frame_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    required = {'clip_id', 'label', 'frame_idx'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")
    return df


def clip_feature_matrix(df_clip: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    # df_clip expected sorted by frame_idx
    # Select landmark columns automatically (lx*,ly*,lz*, rx*,ry*,rz*)
    lm_cols = [c for c in df_clip.columns if c.startswith(('lx', 'ly', 'lz', 'rx', 'ry', 'rz'))]
    if not lm_cols:
        raise ValueError('No landmark columns found in clip dataframe')
    fm = df_clip[lm_cols].to_numpy(dtype=np.float32)
    # fill nan with 0.0
    fm = np.nan_to_num(fm, nan=0.0)
    frame_idxs = df_clip['frame_idx'].astype(int).to_list()
    return fm, frame_idxs


def sliding_windows(features: np.ndarray, frame_idxs: List[int], window_size: int = 16, stride: int = 4, pad: bool = True) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (window, start_frame_idx) windows from features.

    If pad=True, pad start with zeros so the first window is aligned at frame 0.
    """
    T, D = features.shape
    if pad and T < window_size:
        pad_amt = window_size - T
        pad_arr = np.zeros((pad_amt, D), dtype=features.dtype)
        features = np.vstack([pad_arr, features])
        T = features.shape[0]
        # shifted frame index: negative for padded
        frame_idxs = list(range(-pad_amt, 0)) + frame_idxs

    for start in range(0, T - window_size + 1, stride):
        win = features[start:start + window_size]
        yield win, frame_idxs[start] if start < len(frame_idxs) else 0


def generate_windows_from_csv(path: str, window_size: int = 16, stride: int = 4, pad: bool = True) -> Iterator[Dict]:
    """Iterate over all clips in CSV and yield dicts: {'clip_id','label','window','start_frame'}"""
    df = load_frame_csv(path)
    for clip_id, group in df.groupby('clip_id'):
        group_sorted = group.sort_values('frame_idx')
        label = group_sorted['label'].iloc[0]
        features, frame_idxs = clip_feature_matrix(group_sorted)
        for win, start in sliding_windows(features, frame_idxs, window_size=window_size, stride=stride, pad=pad):
            yield {'clip_id': clip_id, 'label': label, 'window': win, 'start_frame': start}


def windows_to_numpy(path: str, window_size: int = 16, stride: int = 4, pad: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert all windows to (N, L, D) numpy array and labels array.

    Returns X (N, L, D), y (N,), clip_ids list parallel to rows (N,)
    """
    Xs = []
    ys = []
    clips = []
    for item in generate_windows_from_csv(path, window_size=window_size, stride=stride, pad=pad):
        Xs.append(item['window'])
        ys.append(item['label'])
        clips.append(item['clip_id'])
    if not Xs:
        return np.zeros((0, window_size, 0)), np.array([]), []
    X = np.stack(Xs, axis=0)
    y = np.array(ys)
    return X, y, clips


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    args = parser.parse_args()
    X, y, clips = windows_to_numpy(args.csv, window_size=args.window, stride=args.stride)
    print('Created windows:', X.shape, 'labels:', y.shape)
