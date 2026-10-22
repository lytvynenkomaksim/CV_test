"""Sample frames for a hand-checked buoy test set, restricted to the zones that matter.

Zone (ROI) per frame = the water band around the skier's depth: rows from 1 skier-height above the ski
down to 4 skier-heights below it (closer to the boat), full width, split into LEFT / MIDDLE / RIGHT thirds.
Buoys in the background (far shore, beyond the skier) are ignored.

    python tools/buoy_gold_sample.py --out data/buoy_gold --per-video 4
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import course4 as C4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--out", default=str(ROOT / "data/buoy_gold"))
ap.add_argument("--per-video", type=int, default=4)
a = ap.parse_args()
out = Path(a.out)
(out / "images").mkdir(parents=True, exist_ok=True)
random.seed(0)


def roi(r, W, H):
    x, y, h = C4.ski_point(r)
    return [0, int(max(0, y - 1.0 * h)), W, int(min(H, y + 4.0 * h))]


items = []
for js in sorted(Path(a.run).glob("*.frames.json")):
    stem = js.name[:-len(".frames.json")]
    meta = json.loads((js.parent / f"{stem}.json").read_text())["meta"]
    W, H = meta["width"], meta["height"]
    frames = json.loads(js.read_text())
    ok = [f for f, r in enumerate(frames) if r["skier"] and (r["skier"][3] - r["skier"][1]) > 20]
    if not ok:
        continue
    # half the frames where the current detector sees something in the zone (to have buoys), half random
    with_b = [f for f in ok if any(roi(frames[f], W, H)[1] <= b[3] <= roi(frames[f], W, H)[3]
                                   for b in (frames[f]["buoys"] or []))]
    pick = set(random.sample(with_b, min(len(with_b), a.per_video // 2 + a.per_video % 2)))
    rest = [f for f in ok if f not in pick and all(abs(f - p) > 30 for p in pick)]
    pick |= set(random.sample(rest, min(len(rest), a.per_video - len(pick))))
    video = ROOT / meta["video"].split("CV_test/")[-1]
    cap = cv2.VideoCapture(str(video))
    for f in sorted(pick):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        okr, img = cap.read()
        if not okr:
            continue
        name = f"{stem[:40]}_f{f:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append(dict(image=f"images/{name}.jpg", video=str(video.relative_to(ROOT)), frame=f,
                          W=W, H=H, skier=frames[f]["skier"], roi=roi(frames[f], W, H)))
(out / "items.json").write_text(json.dumps(items, indent=1))
print(len(items), "frames")
