"""Expand the WLASL subset by selecting video_ids for configured keywords.

Behavior:
- For each keyword, gather video_ids from WLASL_v0.3.json.
- If local video file exists under `Dataset/WLASL/videos/<id>.mp4`, use it.
- If missing and a 'Dataset/missing_videos.txt' file exists listing allowed download ids, and yt-dlp is installed, download it.
- Copy up to TARGET_PER_CLASS videos per keyword into `Dataset/WLASL_subset/<keyword>/`.
"""
import os
import json
from pathlib import Path
import shutil
import subprocess
from collections import defaultdict

from Code import config


def load_wlasl():
    with open(config.WLASL_JSON, 'r') as f:
        return json.load(f)


def map_gloss_to_videoids():
    data = load_wlasl()
    mapping = defaultdict(list)
    for entry in data:
        gloss = entry.get('gloss', '').lower()
        for inst in entry.get('instances', []):
            vid = inst.get('video_id')
            url = inst.get('url')
            if vid:
                mapping[gloss].append({'video_id': vid, 'url': url})
    return mapping


def allowed_to_download(vid):
    # If missing list exists, only allow downloads for listed ids
    missing_file = Path('Dataset/missing_videos.txt')
    if not missing_file.exists():
        return True
    ids = set(l.strip() for l in missing_file.read_text().splitlines() if l.strip())
    return vid in ids


def download_video(vid, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vid}.mp4"
    if out_path.exists():
        return str(out_path)
    try:
        # attempt to use yt_dlp python API if available to fetch by url guessed from vid is not reliable
        import yt_dlp
        ydl_opts = {
            'outtmpl': str(out_path),
            'format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        # We don't have a direct URL here; do not attempt a bad youtube link. Return None so caller can try JSON url.
        return None
    except Exception:
        return None


def download_url_to_path(url, vid, out_dir):
    """Download given URL to out_dir/<vid>.mp4 using yt-dlp where possible, else requests fallback."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vid}.mp4"
    if out_path.exists():
        return str(out_path)
    # skip SWF links (not directly downloadable)
    if url and url.lower().endswith('.swf'):
        return None

    # early skip for known hosts that block automated downloads or have moved
    try:
        from urllib.parse import urlparse
        SKIP_HOSTS = ['handspeak.com', 'aslpro.com', 'aslsignbank.haskins.yale.edu']
        if url:
            netloc = urlparse(url).netloc.lower()
            for h in SKIP_HOSTS:
                if h in netloc:
                    print(f'Skipping download for host {netloc} (configured skip)')
                    return None
    except Exception:
        pass

    # try yt_dlp python API which handles many sources
    try:
        import yt_dlp
        ydl_opts = {
            'outtmpl': str(out_path),
            'format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if out_path.exists():
                return str(out_path)
        except Exception as e:
            print('yt_dlp python API failed for', url, e)
    except Exception:
        # yt_dlp not installed as module; fall back to requests below
        pass
    # fallback to streaming download via requests
    # fallback to streaming download via requests with headers and retries
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
            'Referer': url
        }
        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=30, headers=headers)
                if r.status_code == 200:
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return str(out_path)
                else:
                    print(f'HTTP fallback status {r.status_code} for {url} (attempt {attempt+1})')
            except Exception as e:
                print(f'HTTP fallback error for {url} (attempt {attempt+1}):', e)
    except Exception as e:
        print('requests not available or failed to import:', e)
    return None


def expand(keywords=None, target_per_class=None, download_missing=None, max_total=None, max_per_class=None):
    if keywords is None:
        keywords = config.KEYWORDS
    if target_per_class is None:
        target_per_class = config.TARGET_PER_CLASS
    if download_missing is None:
        download_missing = config.DOWNLOAD_MISSING
    if max_total is None:
        max_total = config.MAX_DOWNLOADS_TOTAL
    if max_per_class is None:
        max_per_class = config.MAX_PER_CLASS_DOWNLOAD
    mapping = map_gloss_to_videoids()
    local_dir = Path(config.WLASL_VIDEOS_DIR)
    subset_dir = Path(config.SUBSET_DIR)
    subset_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    total_downloads = 0
    for kw in keywords:
        insts = mapping.get(kw.lower(), [])
        copied = []
        dst_dir = subset_dir / kw.lower()
        dst_dir.mkdir(parents=True, exist_ok=True)
        for inst in insts:
            if len(copied) >= target_per_class:
                break
            vid = inst.get('video_id')
            url = inst.get('url')
            src = local_dir / f"{vid}.mp4"
            if src.exists():
                dst = dst_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied.append(str(dst))
            else:
                # only attempt download if id listed in missing_videos.txt (allowed_to_download)
                if download_missing and total_downloads < max_total and len(copied) < max_per_class and allowed_to_download(vid):
                    # prefer downloading from the instance URL in JSON
                    d = None
                    if url:
                        d = download_url_to_path(url, vid, local_dir)
                    # fallback to youtube id based download
                    if not d:
                        d = download_video(vid, local_dir)
                    if d:
                        dst = dst_dir / f"{vid}.mp4"
                        try:
                            if not dst.exists():
                                shutil.copy2(d, dst)
                        except Exception:
                            pass
                        copied.append(str(dst))
                        total_downloads += 1
        report[kw] = len(copied)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', nargs='*')
    args = parser.parse_args()
    kw = args.keywords if args.keywords else None
    print('Expanding for keywords:', kw or 'default')
    r = expand(kw)
    print(r)
