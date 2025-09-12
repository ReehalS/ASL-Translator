"""Generate Dataset/missing_videos.txt containing missing video ids whose JSON instance URL is YouTube.

This finds, for configured `Code.config.KEYWORDS`, instances where the instance URL contains 'youtube' or 'youtu.be' and the local file Dataset/WLASL/videos/<id>.mp4 is missing.
It writes unique ids to Dataset/missing_videos.txt.
"""
import sys, os
sys.path.insert(0, os.getcwd())
from Code import config
from Code.wlasl_expand_and_download import map_gloss_to_videoids
from pathlib import Path

def main():
    mapping = map_gloss_to_videoids()
    videos_dir = Path(config.WLASL_VIDEOS_DIR)
    videos_dir.mkdir(parents=True, exist_ok=True)
    missing = set()
    for kw in config.KEYWORDS:
        insts = mapping.get(kw.lower(), [])
        for inst in insts:
            vid = inst.get('video_id')
            url = inst.get('url') or ''
            if not vid:
                continue
            if (videos_dir / f'{vid}.mp4').exists():
                continue
            if 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                missing.add(vid)
    outp = Path('Dataset/missing_videos.txt')
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('\n'.join(sorted(missing)))
    print('Wrote', len(missing), 'YouTube-hosted missing ids to', outp)

if __name__ == '__main__':
    main()
