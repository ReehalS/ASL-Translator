"""Select WLASL videos for given glosses and copy local mp4s into Dataset/WLASL_subset/<gloss>/"""
import json
import os
from pathlib import Path

WLASL_JSON = Path('Dataset/WLASL/WLASL_v0.3.json')
WLASL_VIDEOS = Path('Dataset/WLASL/videos')
DEST = Path('Dataset/WLASL_subset')


def load_json():
    with open(WLASL_JSON, 'r') as f:
        return json.load(f)


def find_video_ids_for_glosses(glosses):
    data = load_json()
    mapping = {g: [] for g in glosses}
    for entry in data:
        g = entry.get('gloss', '').lower()
        if g in mapping:
            for inst in entry.get('instances', []):
                vid = inst.get('video_id')
                if vid:
                    mapping[g].append(vid)
    return mapping


def copy_local_videos(mapping):
    os.makedirs(DEST, exist_ok=True)
    copied = {g: [] for g in mapping}
    for g, vids in mapping.items():
        outdir = DEST / g
        outdir.mkdir(exist_ok=True)
        for v in set(vids):
            fname = WLASL_VIDEOS / f"{v}.mp4"
            if fname.exists():
                dst = outdir / f"{v}.mp4"
                if not dst.exists():
                    import shutil
                    shutil.copy2(fname, dst)
                copied[g].append(str(dst))
    return copied


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('glosses', nargs='+')
    args = parser.parse_args()
    glosses = [g.lower() for g in args.glosses]
    mapping = find_video_ids_for_glosses(glosses)
    copied = copy_local_videos(mapping)
    for g, files in copied.items():
        print(g, len(files))
