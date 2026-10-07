"""Re-create the images of a label-only dataset (e.g. data/buoy_dataset_v2) from the videos.

The dataset keeps labels + manifest.csv (image, split, video, frame) in git; images are cut from
the source videos on demand, so the repo stays small:

    python tools/extract_label_frames.py --dataset data/buoy_dataset_v2
"""
import argparse
import csv
import collections
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(ROOT / "data/buoy_dataset_v2"))
a = ap.parse_args()
ds = Path(a.dataset)
rows = list(csv.DictReader(open(ds / "manifest.csv")))
by_video = collections.defaultdict(list)
for r in rows:
    by_video[r["video"]].append(r)
done = 0
with tqdm(total=len(rows), desc="frames", unit="img") as bar:
    for video, rs in by_video.items():
        vp = ROOT / video
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {vp}")
        for r in sorted(rs, key=lambda r: int(r["frame"])):
            out = ds / "images" / r["split"] / f"{r['image']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
                ok, img = cap.read()
                if not ok:
                    raise SystemExit(f"cannot read frame {r['frame']} of {vp}")
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                done += 1
            bar.update(1)
        cap.release()
print(f"extracted {done} images into {ds / 'images'}")
