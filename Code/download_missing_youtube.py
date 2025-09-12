"""Download missing video ids listed in Dataset/missing_videos.txt only when JSON url is YouTube.

Uses yt_dlp Python API to download YouTube links and copies files into subset per gloss.
"""
import os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
from Code.wlasl_expand_and_download import load_wlasl
import json

def read_missing(path='Dataset/missing_videos.txt'):
    p = Path(path)
    if not p.exists():
        print('missing_videos.txt not found')
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]

def build_index():
    data = load_wlasl()
    idx = {}
    for entry in data:
        gloss = entry.get('gloss','').lower()
        for inst in entry.get('instances',[]):
            vid = inst.get('video_id')
            url = inst.get('url')
            if vid:
                idx.setdefault(vid, []).append({'gloss': gloss, 'url': url})
    return idx

def main():
    ids = read_missing()
    if not ids:
        return
    idx = build_index()
    videos_dir = Path('Dataset/WLASL/videos')
    videos_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = Path('Dataset/WLASL_subset')
    subset_dir.mkdir(parents=True, exist_ok=True)
    success = []
    failed = []
    for vid in ids:
        insts = idx.get(vid)
        if not insts:
            failed.append((vid, 'no-json-entry'))
            continue
        # find a youtube URL
        yt_url = None
        gloss = insts[0].get('gloss') or 'unknown'
        for inst in insts:
            url = inst.get('url')
            if not url:
                continue
            if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                yt_url = url
                break
        if not yt_url:
            failed.append((vid, 'no-youtube-url'))
            continue
        out_path = videos_dir / f'{vid}.mp4'
        if out_path.exists():
            print('already exists', vid)
            success.append(vid)
            # ensure subset copy
            dst = subset_dir / gloss / f'{vid}.mp4'
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                import shutil
                shutil.copy2(out_path, dst)
            continue
        print('Downloading', vid, 'from', yt_url)
        try:
            import yt_dlp
            ydl_opts = {'outtmpl': str(out_path), 'format': 'mp4', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_url])
            if out_path.exists():
                # copy into subset
                dst = subset_dir / gloss / f'{vid}.mp4'
                dst.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(out_path, dst)
                print('  saved', out_path)
                success.append(vid)
            else:
                print('  download finished but file missing for', vid)
                failed.append((vid, 'no-file'))
        except Exception as e:
            print('  download error for', vid, e)
            failed.append((vid, str(e)))

    print('Done. successes:', len(success), 'failures:', len(failed))
    if failed:
        print('Failed items (sample):', failed[:10])

if __name__ == '__main__':
    main()
