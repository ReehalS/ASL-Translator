"""Generate Dataset/missing_videos.txt containing video ids missing from Dataset/WLASL/videos
for the configured keywords.
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
    missing = []
    for kw in config.KEYWORDS:
        insts = mapping.get(kw.lower(), [])
        for inst in insts:
            vid = inst.get('video_id')
            if not vid:
                continue
            p = videos_dir / f'{vid}.mp4'
            if not p.exists():
                missing.append(vid)
    missing = sorted(set(missing))
    outp = Path('Dataset/missing_videos.txt')
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text('\n'.join(missing))
    print('Wrote', len(missing), 'missing ids to', outp)

if __name__ == '__main__':
    main()
