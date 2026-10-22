"""Measure the multi-frame buoy detector on the hand-checked frames: for every test frame, run the clip
[f - before, f + after] at full resolution (pan, skier zone, detectors, local tracking), then score the
buoys reported at frame f exactly like the single-frame detectors (tools/zeroshot_score.py).

    python tools/eval_temporal.py --gold data/buoy_gold_hq --out results/zeroshot_hq/preds [--limit 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import buoys as B  # noqa: E402
from cvdemo import geometry as G  # noqa: E402
from cvdemo import pipeline as PL  # noqa: E402
from cvdemo.colorblob import ColorBlobBuoy, static_overlay_mask  # noqa: E402
from cvdemo.temporal import ScaledDetector as Scaled, TemporalBuoys  # noqa: E402
from cvdemo.verify import ClipVerifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=str(ROOT / "data/buoy_gold_hq"))
ap.add_argument("--out", default=str(ROOT / "results/zeroshot_hq/preds"))
ap.add_argument("--before", type=int, default=12)
ap.add_argument("--after", type=int, default=12)
ap.add_argument("--global-every", type=int, default=2)
ap.add_argument("--limit", type=int)
ap.add_argument("--name", default="temporal")
a = ap.parse_args()
gold = Path(a.gold)
items = json.loads((gold / "items.json").read_text())[: a.limit]


yoloe = B.YoloEZoneBuoy(conf=0.03, imgsz=1920)  # full resolution
cb = ColorBlobBuoy(verifier=ClipVerifier())
skier = PL.SkierTracker("s", imgsz=None)
preds, ts, n_frames = {}, [], 0
masks = {}
for it in tqdm(items, desc="clips"):
    video = ROOT / it["video"]
    if it["video"] not in masks:
        masks[it["video"]] = static_overlay_mask(video)
    cb.overlay = masks[it["video"]]
    tb = TemporalBuoys([Scaled(yoloe, 3.0), cb], global_every=a.global_every)
    pe = G.FlowPanEstimator()
    skier.reset()
    cap = cv2.VideoCapture(str(video))
    f0 = max(0, it["frame"] - a.before)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    t0 = time.perf_counter()
    for f in range(f0, it["frame"] + a.after + 1):
        ok, img = cap.read()
        if not ok:
            break
        best, persons = skier(img)
        pan = pe.update(img, [best[0]] if best else [])
        tb.update(f, img, pan, skier_box=best[0] if best else None, persons=persons)
        n_frames += 1
    ts.append(time.perf_counter() - t0)
    res = tb.result()
    preds[it["image"]] = [list(map(float, b[:5])) for b in res.get(it["frame"], [])]
out = Path(a.out)
(out / f"{a.name}.json").write_text(json.dumps(preds))
tf = out / "timing.json"
tm = json.loads(tf.read_text()) if tf.exists() else {}
tm[a.name] = round(1000 * sum(ts) / max(1, n_frames), 1)  # ms per video frame
tf.write_text(json.dumps(tm, indent=1))
print("ms per frame", tm[a.name], "frames", n_frames)
