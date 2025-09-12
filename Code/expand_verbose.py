"""Run expand with verbose logging to stdout for debugging downloads."""
import sys, os
sys.path.insert(0, os.getcwd())
from Code import config
from Code.wlasl_expand_and_download import map_gloss_to_videoids, download_url_to_path, download_video, expand
from pathlib import Path
import shutil

def verbose_expand():
    mapping = map_gloss_to_videoids()
    report = {}
    total = 0
    for kw in config.KEYWORDS:
        insts = mapping.get(kw.lower(), [])
        print(f'Keyword {kw} has {len(insts)} instances in JSON')
        copied = []
        dst_dir = Path(config.SUBSET_DIR) / kw.lower()
        dst_dir.mkdir(parents=True, exist_ok=True)
        for inst in insts:
            if len(copied) >= config.TARGET_PER_CLASS:
                break
            vid = inst.get('video_id')
            url = inst.get('url')
            src = Path(config.WLASL_VIDEOS_DIR) / f'{vid}.mp4'
            if src.exists():
                dst = dst_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied.append(str(dst))
                print('  COPIED existing', vid)
            else:
                if vid and Path('Dataset/missing_videos.txt').exists():
                    ids = set(l.strip() for l in open('Dataset/missing_videos.txt').read().splitlines() if l.strip())
                else:
                    ids = None
                allow = (ids is None) or (vid in ids)
                if allow and total < config.MAX_DOWNLOADS_TOTAL:
                    print('  ATTEMPT download for', vid, 'url=', url)
                    d = None
                    if url:
                        d = download_url_to_path(url, vid, config.WLASL_VIDEOS_DIR)
                    if not d:
                        d = download_video(vid, config.WLASL_VIDEOS_DIR)
                    if d:
                        dst = dst_dir / f'{vid}.mp4'
                        try:
                            if not dst.exists():
                                shutil.copy2(d, dst)
                        except Exception:
                            pass
                        copied.append(str(dst))
                        total += 1
                        print('    DOWNLOADED', vid)
                    else:
                        print('    FAILED to download', vid)
        report[kw] = len(copied)
    print('Final report:', report)

if __name__ == '__main__':
    verbose_expand()
