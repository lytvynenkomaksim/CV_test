"""Combine buoy detectors: union of candidates (old trained model, YOLOE 'buoy', colour centre-surround), each
re-checked by CLIP on an enlarged crop; final score = a vote over the detectors' scores + CLIP.

    python tools/ensemble_buoys.py --gold data/buoy_gold_hq --preds results/zeroshot_hq/preds
Writes <preds>/ensemble_*.json so tools/zeroshot_score.py scores them like any detector."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo.verify import ClipVerifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=str(ROOT / "data/buoy_gold_hq"))
ap.add_argument("--preds", default=str(ROOT / "results/zeroshot_hq/preds"))
a = ap.parse_args()
gold, pdir = Path(a.gold), Path(a.preds)
items = json.loads((gold / "items.json").read_text())
SRC = {"finetuned": 0.05, "yoloe26l": 0.03, "colorblob": 0.0}  # minimum score to become a candidate
P = {k: json.loads((pdir / f"{k}.json").read_text()) for k in SRC}
clip = ClipVerifier()
out = {k: {} for k in ("ensemble_vote", "ensemble_untrained", "ensemble_clip_only")}
ts = []
for it in items:
    im = it["image"]
    img = cv2.imread(str(gold / im))
    x0, y0, x1, y1 = it["roi"]
    cands = []  # [box, {src: score}]
    for src, thr in SRC.items():
        for b in P[src].get(im, []):
            if b[4] < thr or not (y0 <= (b[1] + b[3]) / 2 <= y1) or b[2] - b[0] > 80 or b[3] - b[1] > 80:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            m = next((c for c in cands if abs((c[0][0] + c[0][2]) / 2 - cx) < max(6, 0.7 * (c[0][2] - c[0][0]))
                      and abs((c[0][1] + c[0][3]) / 2 - cy) < max(6, 0.7 * (c[0][3] - c[0][1]))), None)
            if m is None:
                cands.append([list(b[:4]), {src: b[4]}])
            else:
                m[1][src] = max(m[1].get(src, 0), b[4])
    t = time.perf_counter()
    pc = clip(img, [c[0] for c in cands]) if cands else []
    ts.append(time.perf_counter() - t)
    for c, p in zip(cands, pc):
        s = c[1]
        ft, ye, cb = s.get("finetuned", 0), min(1, 3 * s.get("yoloe26l", 0)), 1.0 if "colorblob" in s else 0.0
        out["ensemble_vote"].setdefault(im, []).append([*c[0], float((ft + ye + 0.5 * cb + 2 * p) / 4.5)])
        out["ensemble_untrained"].setdefault(im, []).append([*c[0], float((ye + 0.5 * cb + 2 * p) / 3.5)]
                                                            if (ye > 0 or cb > 0) else [*c[0], 0.0])
        out["ensemble_clip_only"].setdefault(im, []).append([*c[0], float(p)])
for k, v in out.items():
    (pdir / f"{k}.json").write_text(json.dumps(v))
tf = pdir / "timing.json"
tm = json.loads(tf.read_text())
base = {k: tm.get(k, 0) for k in SRC}
clip_ms = 1000 * float(np.median(ts))
tm.update(ensemble_vote=round(sum(base.values()) + clip_ms, 1),
          ensemble_untrained=round(base["yoloe26l"] + base["colorblob"] + clip_ms, 1),
          ensemble_clip_only=round(sum(base.values()) + clip_ms, 1))
tf.write_text(json.dumps(tm, indent=1))
print("CLIP ms per frame", round(clip_ms))
