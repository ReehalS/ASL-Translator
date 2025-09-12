"""Download missing WLASL video IDs listed in Dataset/missing_videos.txt.

This script:
 - reads Dataset/missing_videos.txt for one video id per line
 - looks up each id in Dataset/WLASL/WLASL_v0.3.json to find the instance URL and gloss
 - attempts to download using yt-dlp first, then HTTP streaming fallback
 - saves file to Dataset/WLASL/videos/<id>.mp4 and copies it into Dataset/WLASL_subset/<gloss>/
"""
import json
import os
import sys
import time
from pathlib import Path
import shutil

sys.path.insert(0, os.getcwd())
from Code.wlasl_expand_and_download import load_wlasl, download_url_to_path, download_video


def read_missing_ids(path='Dataset/missing_videos.txt'):
    p = Path(path)
    if not p.exists():
        print('No missing_videos.txt found at', path)
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def build_index(json_path='Dataset/WLASL/WLASL_v0.3.json'):
    data = load_wlasl()
    # map video_id -> list of instances (url, gloss)
    idx = {}
    for entry in data:
        gloss = entry.get('gloss', '').lower()
        for inst in entry.get('instances', []):
            vid = inst.get('video_id')
            url = inst.get('url')
            if not vid:
                continue
            idx.setdefault(vid, []).append({'gloss': gloss, 'url': url})
    return idx


def main():
    ids = read_missing_ids()
    if not ids:
        return
    idx = build_index()
    videos_dir = Path('Dataset/WLASL/videos')
    videos_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = Path('Dataset/WLASL_subset')
    subset_dir.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        insts = idx.get(vid)
        if not insts:
            print('ID not found in JSON:', vid)
            continue
        # try every instance url for this video id, preferring YouTube links
        out = None
        gloss = insts[0].get('gloss') or 'unknown'
        print('Downloading', vid, 'for gloss', gloss)
        # sort instances so youtube links come first
        insts_sorted = sorted(insts, key=lambda x: 0 if x.get('url') and 'youtube' in x.get('url').lower() else 1)
        for inst in insts_sorted:
            url = inst.get('url')
            if not url:
                continue
            print('  trying URL:', url)
            try:
                # try yt_dlp python API for any URL
                try:
                    import yt_dlp
                    ydl_opts = {'outtmpl': str(videos_dir / f'{vid}.mp4'), 'format': 'mp4', 'quiet': True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    if (videos_dir / f'{vid}.mp4').exists():
                        out = str(videos_dir / f'{vid}.mp4')
                        print('    yt_dlp succeeded')
                        break
                except Exception as e:
                    print('    yt_dlp failed for url:', e)
                # fallback to download_url_to_path which tries HTTP
                out = download_url_to_path(url, vid, videos_dir)
                if out:
                    print('    HTTP download succeeded')
                    break
            except Exception as e:
                print('    error trying url', e)
        if not out:
            print('  fallback to generic video id download (not preferred) for', vid)
            out = download_video(vid, videos_dir)
        if out:
            # copy into subset gloss folder
            dst_folder = subset_dir / gloss
            dst_folder.mkdir(parents=True, exist_ok=True)
            dst_path = dst_folder / f'{vid}.mp4'
            try:
                if not dst_path.exists():
                    shutil.copy2(out, dst_path)
                print('  saved to', out, 'and copied to subset', dst_path)
            except Exception as e:
                print('  copy failed:', e)
        else:
            print('  failed to download', vid)
        # small delay
        time.sleep(0.5)


if __name__ == '__main__':
    main()
