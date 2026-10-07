"""Clean Grounding-DINO pseudo labels for the buoy detector (dataset v2).

Works on the raw detections stored in autolabel_meta.jsonl, so it can also RECOVER buoys that the
first auto-labelling dropped (it discarded any buoy inside the skier's box - but the skier often
passes right over / next to the buoy, which is exactly the moment we need).

Rules (each removal is counted and shown in a gallery for review):
  shape     - box too large / too small / too elongated for a buoy
  body      - centre on the skier's body (head, torso, arms from YOLO11-pose); feet / ski area is kept
  own_boat  - inside our own boat (GDINO 'boat' box touching the bottom edge: stern, steering wheel)
  temporal  - weak detection (score < --strong) without a similar box in the neighbouring samples
              (real buoys persist over consecutive samples; spray blobs flicker)

    python tools/clean_dataset.py --src <dataset dir with autolabel_meta.jsonl + images> \
        --out data/buoy_dataset_v2 [--copy-images]
Images are not stored in git: tools/extract_label_frames.py re-creates them from the videos.
"""
import argparse
import collections
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default=str(ROOT / "data/buoy_dataset_v2"))
ap.add_argument("--min-score", type=float, default=0.25, help="lowest Grounding DINO score considered")
ap.add_argument("--strong", type=float, default=0.42, help="kept without temporal support above this score")
ap.add_argument("--copy-images", action="store_true")
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402


src, out = Path(a.src), Path(a.out)
meta = [json.loads(l) for l in open(src / "autolabel_meta.jsonl")]
# newest entry wins if the same image was labelled twice
meta = list({m["image"]: m for m in meta}.values())
pose = YOLO(str(ROOT / "models/yolo11s-pose.pt"))
BODY_KPTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # head, shoulders, elbows, wrists, hips (not knees/ankles)


def img_path(m):
    return src / "images" / m["split"] / f"{m['image']}.jpg"


def body_box(img, persons):
    """Region of the skier's head/torso/arms (feet and ski excluded)."""
    regions = []
    H, W = img.shape[:2]
    for p in persons:
        x0, y0, x1, y1 = p
        pw, ph = x1 - x0, y1 - y0
        cx0, cy0 = int(max(0, x0 - 0.3 * pw)), int(max(0, y0 - 0.3 * ph))
        cx1, cy1 = int(min(W, x1 + 0.3 * pw)), int(min(H, y1 + 0.3 * ph))
        crop = img[cy0:cy1, cx0:cx1]
        k = None
        if crop.size:
            r = pose.predict(crop, imgsz=320, conf=0.15, verbose=False)[0]
            if r.keypoints is not None and len(r.keypoints):
                k = r.keypoints.data[int(r.boxes.conf.argmax())].cpu().numpy()
        if k is not None and (k[BODY_KPTS, 2] > 0.3).sum() >= 4:
            pts = k[BODY_KPTS][k[BODY_KPTS, 2] > 0.3, :2] + [cx0, cy0]
            bx0, by0 = pts.min(0)
            bx1, by1 = pts.max(0)
            m = 0.12 * max(bx1 - bx0, by1 - by0)
            regions.append((bx0 - m, by0 - m, bx1 + m, by1 + m))
        else:  # no pose: upper 65 % of the person box
            regions.append((x0, y0, x1, y0 + 0.65 * ph))
    return regions


def buoy_colour(img, b):
    """Buoys are bright, saturated yellow / green-yellow, red or pink; boots, skin, the ski and spray are not."""
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    p = cv2.cvtColor(img[max(0, y0):y1 + 1, max(0, x0):x1 + 1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    if len(p) < 4:
        return False
    p = p[np.argsort(p[:, 1] * p[:, 2])[-max(3, len(p) // 4):]]  # most colourful quarter
    h, sat, val = np.median(p[:, 0]), np.median(p[:, 1]), np.median(p[:, 2])
    hue_ok = 22 <= h <= 48 or h >= 150 or h <= 4
    return bool(hue_ok and sat >= 110 and val >= 120)


def inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# ---- pass 1: per-image candidates + single-frame rules
cands = {}
reasons = collections.Counter()
removed = collections.defaultdict(list)  # rule -> [(image, box)]
recovered = []
for m in tqdm(meta, desc="images (pose + rules)"):
    img = cv2.imread(str(img_path(m)))
    if img is None:
        continue
    H, W = img.shape[:2]
    dets = m["all"]
    persons = [d[2:] for d in dets if d[0] == "person" and d[1] > 0.3]
    own_boat = [d[2:] for d in dets if "boat" in d[0] and d[1] > 0.3 and d[5] >= 0.97 * H]
    bodies = body_box(img, persons) if persons else []
    old = {tuple(round(v, 1) for v in b[1:]) for b in m["buoys"]}
    keep = []
    for d in dets:
        if d[0] != "buoy" or d[1] < a.min_score:
            continue
        s, b = d[1], d[2:]
        w, h = b[2] - b[0], b[3] - b[1]
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        rule = None
        if w > 0.06 * W or h > 0.08 * H or w * h < 9 or not (0.35 <= w / max(h, 1) <= 2.8):
            rule = "shape"
        elif any(inside(c, r) for r in bodies):
            rule = "body"
        elif any(inside(c, r) for r in own_boat):
            rule = "own_boat"
        elif any(inside(c, (p[0] - 0.3 * (p[2] - p[0]), p[1], p[2] + 0.3 * (p[2] - p[0]), p[3] + 0.3 * (p[3] - p[1])))
                 for p in persons) and not buoy_colour(img, b):
            rule = "skier_legs_ski"  # near the skier only clearly buoy-coloured blobs are kept
        if rule:
            reasons[rule] += 1
            removed[rule].append((m["image"], m["split"], b))
            continue
        keep.append((s, b))
        if tuple(round(v, 1) for v in b) not in old and s >= 0.28 and persons:
            recovered.append((m["image"], m["split"], b))
    cands[m["image"]] = dict(meta=m, W=W, H=H, boxes=keep)

# ---- pass 2: temporal support within each video
by_video = collections.defaultdict(list)
for k, c in cands.items():
    by_video[c["meta"]["video"]].append(k)
final = {}
for v, keys in by_video.items():
    keys.sort(key=lambda k: cands[k]["meta"]["frame"])
    for i, k in enumerate(keys):
        c = cands[k]
        W = c["W"]
        nbrs = [cands[keys[j]] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(keys)
                and abs(cands[keys[j]]["meta"]["frame"] - c["meta"]["frame"]) <= 15]
        kept = []
        for s, b in c["boxes"]:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            area = (b[2] - b[0]) * (b[3] - b[1])
            support = any(np.hypot((nb[0] + nb[2]) / 2 - cx, (nb[1] + nb[3]) / 2 - cy) < 0.12 * W
                          and 0.4 < ((nb[2] - nb[0]) * (nb[3] - nb[1])) / max(area, 1) < 2.5
                          for n in nbrs for _, nb in n["boxes"])
            if s >= a.strong or support:
                kept.append((s, b))
            else:
                reasons["temporal"] += 1
                removed["temporal"].append((k, c["meta"]["split"], b))
        final[k] = kept

# ---- write dataset v2 (labels + manifest; images optional)
for sp in ("train", "val"):
    (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    if a.copy_images:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
rows, n_boxes = [], collections.Counter()
for k, boxes in final.items():
    c = cands[k]
    m, W, H = c["meta"], c["W"], c["H"]
    with open(out / "labels" / m["split"] / f"{k}.txt", "w") as f:
        for s, b in boxes:
            f.write(f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} "
                    f"{(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n")
    n_boxes[m["split"]] += len(boxes)
    rows.append(dict(image=k, split=m["split"], video=m["video"], frame=m["frame"], boxes=len(boxes)))
    if a.copy_images:
        shutil.copy(img_path(m), out / "images" / m["split"] / f"{k}.jpg")
with open(out / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["video"], r["frame"])))
(out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: buoy\n")
old_counts = collections.Counter()
for m in meta:
    old_counts[m["split"]] += len(m["buoys"])
stats = dict(images=len(final), boxes_v1=dict(old_counts), boxes_v2=dict(n_boxes), removed=dict(reasons),
             recovered_near_skier=len(recovered), params=dict(min_score=a.min_score, strong=a.strong))
(out / "cleaning_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))

# ---- galleries for review
random.seed(0)
gal = out / "review"
gal.mkdir(exist_ok=True)


def gallery(items, name, n=60):
    tiles = []
    for img_name, split, b in random.sample(items, min(n, len(items))):
        img = cv2.imread(str(src / "images" / split / f"{img_name}.jpg"))
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        r = int(max(24, 1.5 * max(b[2] - b[0], b[3] - b[1])))
        crop = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_NEAREST)
        tiles.append(crop)
    if not tiles:
        return
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(gal / f"{name}.jpg"), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


for rule, items in removed.items():
    gallery(items, f"removed_{rule}")
gallery(recovered, "recovered_near_skier")
kept_items = [(k, cands[k]["meta"]["split"], b) for k, bs in final.items() for _, b in bs]
gallery(kept_items, "kept_random")
