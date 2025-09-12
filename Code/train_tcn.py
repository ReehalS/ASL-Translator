"""Train a small TCN on windowed landmark data.
"""
import os
import random
import joblib
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from Code.dataset.windowed_generator import windows_to_numpy
from Code.models.tcn import TCN


def label_map(labels):
    uniq = sorted(set(labels))
    lm = {v: i for i, v in enumerate(uniq)}
    return lm


def train(args):
    X, y, clips = windows_to_numpy(args.csv, window_size=args.window, stride=args.stride)
    if X.size == 0:
        print('No windows found in CSV')
        return
    # simple label map
    lm = label_map(y)
    y_idx = np.array([lm[v] for v in y], dtype=np.int64)

    # split train/val by clip id (clip-wise split)
    clip_to_idx = defaultdict(list)
    for i, c in enumerate(clips):
        clip_to_idx[c].append(i)
    clip_ids = list(clip_to_idx.keys())
    random.shuffle(clip_ids)
    split = int(0.8 * len(clip_ids))
    train_clips = set(clip_ids[:split])
    train_idx = [i for i, c in enumerate(clips) if c in train_clips]
    val_idx = [i for i, c in enumerate(clips) if c not in train_clips]

    X_train = X[train_idx]
    y_train = y_idx[train_idx]
    X_val = X[val_idx]
    y_val = y_idx[val_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, L, D = X_train.shape
    num_classes = len(lm)
    model = TCN(input_dim=D, num_classes=num_classes, num_channels=[64, 64], kernel_size=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    batch_size = args.batch
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(X_train_t))
        losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # val
        model.eval()
        with torch.no_grad():
            out_val = model(X_val_t)
            pred = out_val.argmax(dim=1)
            acc = (pred == y_val_t).float().mean().item()
        print(f'Epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f} val_acc={acc:.3f}')

    # save model and label map
    os.makedirs('Models', exist_ok=True)
    save_path = os.path.join('Models', args.out)
    torch.save({'model_state_dict': model.state_dict(), 'label_map': lm}, save_path)
    print('Saved model to', save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', type=str, default='gesture_wlasl_tcn.pt')
    args = parser.parse_args()
    train(args)
