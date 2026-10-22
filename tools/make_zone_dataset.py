"""Small, fast buoy-training set built ONLY from the zone near the skier, with REAL buoys only.

    python tools/make_zone_dataset.py --out data/buoy_zone --paste-per-tile 1

1. Real tiles: frames + cleaned labels from data/buoy_dataset_v2. The zone (water band from 1 skier-height
   beyond the ski to 4 heights closer to the boat) is cut into LEFT / MIDDLE / RIGHT tiles at native
   resolution (small images, buoys relatively big, no shore / trees to learn -> fast).
2. No "missed buoy = background" lessons: anything a high-recall search finds that has no label (the
   fine-tuned detector at confidence 0.05 + strongly coloured blobs), outside the skier, is painted over with
   the surrounding water (inpainting) before training. The model then learns nothing from that spot - neither
   "buoy" nor "not a buoy". The skier and ski are never painted over: they stay as true negatives (a pink ski
   is not a buoy).
3. More real buoys: buoy cut-outs harvested from ALL frames of the training videos (confident detections on
   buoy tracks that last >= 8 frames and move past the skier) are pasted, with their real colour and look,
   onto copies of the tiles (soft-edged blend, size jitter 0.7-1.4x, light blur / JPEG only).
4. Videos used for testing (--hold-out) are never used.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import buoys as B  # noqa: E402
from cvdemo import course4 as C4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data/buoy_dataset_v2"))
ap.add_argument("--frames-dir", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--out", default=str(ROOT / "data/buoy_zone"))
ap.add_argument("--detector", default=str(ROOT / "weights/buoy_yolo11s.pt"), help="for the missed-buoy search")
ap.add_argument("--paste-per-tile", type=int, default=1)
ap.add_argument("--hold-out", nargs="*", default=["Aaron_Davis", "Lucas Cornale 6 at 35", "Vincenzo Marino 6 at 35",
                                                   "youtube"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
src, out = Path(a.src), Path(a.out)


def zone(rec, W, H):
    if rec and rec.get("skier") and rec["skier"][3] - rec["skier"][1] > 20:
        x, y, h = C4.ski_point(rec)
        return int(max(0, y - 1.0 * h)), int(min(H, y + 4.0 * h))
    return int(0.45 * H), H


def read_labels(p, W, H):
    b = []
    if p.exists():
        for line in open(p):
            _, x, y, w, h = map(float, line.split())
            b.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return b


def overlaps(b, others, pad=2):
    return any(not (b[2] < o[0] - pad or b[0] > o[2] + pad or b[3] < o[1] - pad or b[1] > o[3] + pad) for o in others)


def colour_blobs(img):
    """Strongly coloured compact blobs whose hue differs from the water (high recall, many false alarms)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    water = float(np.median(h[s > 40])) if (s > 40).any() else 90.0
    dh = np.minimum(np.abs(h.astype(int) - water), 180 - np.abs(h.astype(int) - water))
    m = ((dh >= 18) & (s >= 110) & (v >= 90)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)))
    return [[x, y, x + w, y + hh] for x, y, w, hh, ar in st[1:] if ar >= 6 and w < 60 and hh < 60
            and 0.3 < w / max(hh, 1) < 3.5]


# ------------------------------------------------------------------ 1. harvest real buoy cut-outs
def looks_like_buoy(c, p):
    """A buoy stands out from the water around it: brighter or more colourful than the ring of pixels
    around it, and not a dark overlay / shadow patch."""
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = c.shape[:2]
    ring = np.ones((h, w), bool)
    ring[p:h - p, p:w - p] = False
    inner = ~ring
    if inner.sum() < 4 or (hsv[..., 2] < 50).mean() > 0.15:
        return False
    dv = hsv[..., 2][inner].mean() - hsv[..., 2][ring].mean()
    ds = hsv[..., 1][inner].mean() - hsv[..., 1][ring].mean()
    return dv > 15 or ds > 30


def water_like(t):
    """Paste target: no dark caption boxes / black overlay inside it."""
    return (cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[..., 2] < 60).mean() < 0.03


(out / "cutouts").mkdir(parents=True, exist_ok=True)
def frames_for(video, cache={}):
    stem = Path(video).stem.replace(" ", "_")
    if stem not in cache:
        p = next((q for q in Path(a.frames_dir).glob("*.frames.json") if q.name.startswith(stem)), None)
        cache[stem] = (json.loads(p.read_text()), json.loads(p.with_name(p.name.replace(".frames", "")).read_text())
                       ["meta"]) if p else (None, None)
    return cache[stem]


rows = [r for r in csv.DictReader(open(src / "manifest.csv")) if not any(h in r["video"] for h in a.hold_out)]
videos = sorted({r["video"] for r in rows})
cutouts = []
for video in tqdm(videos, desc="harvest real buoys"):
    fr, meta = frames_for(video)
    if fr is None:
        continue
    fr = json.loads(json.dumps(fr))
    C4.drop_skier_attached(fr, meta["fps"])
    tr = B.BuoyTracker(max_miss=int(2 * meta["fps"]), frame_size=(meta["width"], meta["height"]))
    for f, r in enumerate(fr):
        if r["buoys"] is not None:
            tr.update(f, r["buoys"], r.get("pan", 0.0))
    want = {}
    for t in tr.all_tracks():
        obs = [h for h in t.hist if h[6]]
        if len(obs) < 8:
            continue
        last = -99
        for h in obs:
            f = h[0]
            if f - last < 5:
                continue
            for b in fr[f]["buoys"] or []:
                cx, by = (b[0] + b[2]) / 2, b[3]
                if b[4] >= 0.35 and b[3] - b[1] >= 7 \
                        and abs(by - h[2]) < 4 and abs(cx + fr[f].get("pan", 0.0) - h[1]) < 4:
                    want.setdefault(f, []).append(b[:4])
                    last = f
                    break
    cap = cv2.VideoCapture(str(ROOT / video))
    for f in sorted(want):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        sk = fr[f]["skier"]
        H, W = img.shape[:2]
        for b in want[f]:
            if sk and overlaps(b, [sk[:4]]):
                continue
            if not (0.12 * H < (b[1] + b[3]) / 2 < 0.85 * H):  # TV caption / score bands
                continue
            x0, y0, x1, y1 = map(int, b)
            p = max(2, int(0.35 * (y1 - y0)))
            c = img[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p]
            if c.size and min(c.shape[:2]) >= 7 and looks_like_buoy(c, p):
                cutouts.append(c.copy())
                if len(cutouts) % 10 == 0:
                    cv2.imwrite(str(out / "cutouts" / f"{len(cutouts):05d}.png"), c)
print(len(cutouts), "real buoy cut-outs harvested")


def paste(tile, c, avoid):
    s = rng.uniform(0.7, 1.4)
    c = cv2.resize(c, (max(5, int(c.shape[1] * s)), max(5, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h, w = c.shape[:2]
    H, W = tile.shape[:2]
    if h >= H or w >= W:
        return None
    for _ in range(15):
        x0, y0 = rng.randint(0, W - w), rng.randint(0, H - h)
        p = int(round(0.35 / 1.7 * h))  # the margin around the buoy in the cut-out
        bb = [x0 + p, y0 + p, x0 + w - p, y0 + h - p]
        if overlaps([x0, y0, x0 + w, y0 + h], avoid, 4) or not water_like(
                tile[max(0, y0 - h):y0 + 2 * h, max(0, x0 - w):x0 + 2 * w]):
            continue
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2 + 0.5) / (w / 2)) ** 2 + ((yy - h / 2 + 0.5) / (h / 2)) ** 2
        al = np.clip((1 - r2) * 2.5, 0, 1)[..., None]  # soft edge: the cut-out's own water fades into the tile
        roi = tile[y0:y0 + h, x0:x0 + w].astype(np.float32)
        tile[y0:y0 + h, x0:x0 + w] = (roi * (1 - al) + c.astype(np.float32) * al).astype(np.uint8)
        return bb
    return None


def save(img, boxes, name, split):
    H, W = img.shape[:2]
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(out / "labels" / split / f"{name}.txt", "w") as f:
        for b in boxes:
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                f.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")


# ------------------------------------------------------------------ 2. tiles, missed-buoy masking, pasting
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO  # noqa: E402

det = YOLO(a.detector)
stats = dict(tiles=0, pasted_tiles=0, real_boxes=0, pasted_boxes=0, masked_unlabelled=0, cutouts=len(cutouts))
val_videos = set(videos[::6])  # whole videos for validation (no leakage between neighbouring frames)
for r in tqdm(rows, desc="tiles"):
    img = cv2.imread(str(src / "images" / r["split"] / f"{r['image']}.jpg"))
    H, W = img.shape[:2]
    fr, _ = frames_for(r["video"])
    rec = fr[int(r["frame"])] if fr and int(r["frame"]) < len(fr) else None
    y0, y1 = zone(rec, W, H)
    if y1 - y0 < 40:
        continue
    boxes = read_labels(src / "labels" / r["split"] / f"{r['image']}.txt", W, H)
    sk = rec["skier"][:4] if rec and rec.get("skier") else None
    band = img[y0:y1].copy()
    # high-recall search for buoys without a label -> inpaint them away (unless on the skier / ski)
    res = det.predict(band, imgsz=1280, conf=0.05, verbose=False)[0]
    cand = [list(map(float, b)) for b in res.boxes.xyxy.tolist()] + colour_blobs(band)
    lab_b = [[b[0], b[1] - y0, b[2], b[3] - y0] for b in boxes]
    sk_b = [[sk[0] - 0.3 * (sk[2] - sk[0]), sk[1] - y0, sk[2] + 0.3 * (sk[2] - sk[0]),
             sk[3] - y0 + 0.25 * (sk[3] - sk[1])]] if sk else []
    mask = np.zeros(band.shape[:2], np.uint8)
    for c in cand:
        if overlaps(c, lab_b, 3) or overlaps(c, sk_b, 0):
            continue
        pad = 3
        cv2.rectangle(mask, (int(c[0]) - pad, int(c[1]) - pad), (int(c[2]) + pad, int(c[3]) + pad), 255, -1)
        stats["masked_unlabelled"] += 1
    if mask.any():
        band = cv2.inpaint(band, mask, 5, cv2.INPAINT_TELEA)
    split = "val" if r["video"] in val_videos else "train"
    for k in range(3):  # LEFT / MIDDLE / RIGHT tiles, small overlap
        x0, x1 = int(max(0, k * W / 3 - 0.03 * W)), int(min(W, (k + 1) * W / 3 + 0.03 * W))
        tile = band[:, x0:x1].copy()
        tb = [[b[0] - x0, b[1], b[2] - x0, b[3]] for b in lab_b if x0 <= (b[0] + b[2]) / 2 < x1]
        name = f"{r['image']}_z{k}"
        save(tile, tb, name, split)
        stats["tiles"] += 1
        stats["real_boxes"] += len(tb)
        if split == "train" and cutouts:
            avoid = tb + ([[sk_b[0][0] - x0, sk_b[0][1], sk_b[0][2] - x0, sk_b[0][3]]] if sk_b else [])
            for j in range(a.paste_per_tile):
                t2, new = tile.copy(), []
                for _ in range(rng.randint(1, 4)):
                    bb = paste(t2, rng.choice(cutouts), avoid + new)
                    if bb:
                        new.append(bb)
                if rng.random() < 0.3:
                    t2 = cv2.GaussianBlur(t2, (3, 3), 0)
                t2 = cv2.imdecode(cv2.imencode(".jpg", t2, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 95)])[1], 1)
                save(t2, tb + new, f"{name}_p{j}", split)
                stats["pasted_tiles"] += 1
                stats["pasted_boxes"] += len(new)
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: buoy\n")
(out / "stats.json").write_text(json.dumps(dict(stats, val_videos=sorted(val_videos), hold_out=a.hold_out), indent=1))
print(stats)
