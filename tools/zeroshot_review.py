"""Review sheets for hand-checking buoys: union of all detectors' candidates in the zone, numbered.

    python tools/zeroshot_review.py --gold data/buoy_gold --preds results/zeroshot/preds --out <dir>
Writes <dir>/sheet_XX.jpg (3 zone crops per sheet, 2x enlarged, candidates numbered, 50 px ruler ticks) and
<dir>/candidates.json {image: [[x0,y0,x1,y1], ...]} in full-frame coordinates; the reviewer then writes
data/buoy_gold/labels.json {image: {"keep": [candidate numbers], "add": [[x0,y0,x1,y1], ...]}}.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default="data/buoy_gold")
ap.add_argument("--preds", default="results/zeroshot/preds")
ap.add_argument("--out", required=True)
ap.add_argument("--min-score", type=float, default=0.2)
a = ap.parse_args()
gold, out = Path(a.gold), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
items = json.loads((gold / "items.json").read_text())
preds = {p.stem: json.loads(p.read_text()) for p in Path(a.preds).glob("*.json") if p.stem != "timing"}

cands = {}
for it in items:
    x0, y0, x1, y1 = it["roi"]
    boxes = []
    for name, P in preds.items():
        for b in P.get(it["image"], []):
            cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
            if b[4] < a.min_score or not (y0 <= cy <= y1) or w > 0.12 * it["W"] or h > 0.2 * it["H"]:
                continue
            m = next((c for c in boxes if abs(c[0] - cx) < max(6, 0.6 * c[2]) and abs(c[1] - cy) < max(6, 0.6 * c[3])), None)
            if m is None:
                boxes.append([cx, cy, w, h, b[:4]])
    cands[it["image"]] = [c[4] for c in boxes]
(out / "candidates.json").write_text(json.dumps(cands))

tiles = []
for it in items:
    img = cv2.imread(str(gold / it["image"]))
    x0, y0, x1, y1 = it["roi"]
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for xt in range(0, x1 - x0, 50):
        cv2.line(crop, (2 * xt, 0), (2 * xt, 10 if xt % 100 else 22), (255, 255, 255), 2)
        if xt % 100 == 0:
            cv2.putText(crop, str(xt + x0), (2 * xt + 3, 34), 0, 0.6, (255, 255, 255), 2)
    for yt in range(0, y1 - y0, 50):
        cv2.line(crop, (0, 2 * yt), (10 if yt % 100 else 22, 2 * yt), (255, 255, 255), 2)
        cv2.putText(crop, str(yt + y0), (24, 2 * yt + 6), 0, 0.6, (255, 255, 255), 2)
    for k, b in enumerate(cands[it["image"]]):
        p0 = (int(2 * (b[0] - x0)) - 4, int(2 * (b[1] - y0)) - 4)
        p1 = (int(2 * (b[2] - x0)) + 4, int(2 * (b[3] - y0)) + 4)
        cv2.rectangle(crop, p0, p1, (0, 255, 255), 1)
        cv2.putText(crop, str(k), (p1[0] + 2, p0[1] + 8), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(crop, str(k), (p1[0] + 2, p0[1] + 8), 0, 0.7, (0, 255, 255), 2)
    title = f"{it['image'][7:-4]}"
    crop = cv2.copyMakeBorder(crop, 36, 4, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(crop, title, (8, 26), 0, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(crop, (1800, int(1800 * crop.shape[0] / crop.shape[1]))))
for s in range(0, len(tiles), 3):
    cv2.imwrite(str(out / f"sheet_{s // 3:02d}.jpg"), np.vstack(tiles[s:s + 3]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(tiles), "tiles,", sum(map(len, cands.values())), "candidates")
