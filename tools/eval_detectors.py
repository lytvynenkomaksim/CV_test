"""Evaluate buoy detectors on the same held-out frames with the same metric.

    python tools/eval_detectors.py --dataset data/buoy_dataset_v2 \
        --model yolo:models/buoy_yolo11s.pt --model yolo:models/buoy_yolo26s.pt \
        --model rfdetr:models/buoy_rfdetr.pth:RFDETRSmall --out results/eval

Metrics per model (val split, held-out videos):
  AP50            - COCO-style average precision at IoU 0.5
  P / R @ best-F1 - precision and recall at the confidence threshold with the best F1
  R@centre        - recall when a hit only needs the predicted centre within max(8 px, 1 box size)
                    of the label (IoU is harsh for 8-15 px objects)
  ms/img          - median latency on this machine (GPU if available)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(ROOT / "data/buoy_dataset_v2"))
ap.add_argument("--split", default="val")
ap.add_argument("--model", action="append", required=True,
                help="yolo:weights[:imgsz] | rfdetr:weights[:Variant[:resolution]] | gdino:-")
ap.add_argument("--out", default=str(ROOT / "results/eval"))
a = ap.parse_args()


def load_labels(p, W, H):
    out = []
    if p.exists():
        for line in open(p):
            _, x, y, w, h = map(float, line.split())
            out.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return out


def iou(a_, b_):
    x0, y0, x1, y1 = max(a_[0], b_[0]), max(a_[1], b_[1]), min(a_[2], b_[2]), min(a_[3], b_[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - inter
    return inter / u if u > 0 else 0


def build(spec):
    kind, w, *extra = spec.split(":")
    if kind == "yolo":
        from ultralytics import YOLO
        m = YOLO(w)
        imgsz = int(extra[0]) if extra else 1280
        return f"{Path(w).stem}@{imgsz}", lambda img: [
            (*b, s) for b, s in zip(*(lambda r: (r.boxes.xyxy.tolist(), r.boxes.conf.tolist()))(
                m.predict(img, imgsz=imgsz, conf=0.01, verbose=False)[0]))]
    if kind == "rfdetr":
        import rfdetr
        kw = dict(pretrain_weights=w)
        if len(extra) > 1:
            kw["resolution"] = int(extra[1])
        m = getattr(rfdetr, extra[0] if extra else "RFDETRSmall")(**kw)
        try:
            m.optimize_for_inference()
        except Exception:
            pass
        return f"{Path(w).stem}-{extra[0] if extra else 'RFDETRSmall'}", lambda img: [
            (*b, float(s)) for b, s in zip(*(lambda d: (d.xyxy.tolist(), d.confidence.tolist()))(
                m.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=0.01)))]
    if kind == "gdino":
        from cvdemo.buoys import GDinoBuoy
        g = GDinoBuoy(thr=0.15)
        return "grounding-dino-tiny (zero-shot)", lambda img: g(img)
    raise ValueError(kind)


ds = Path(a.dataset)
imgs = sorted((ds / "images" / a.split).glob("*.jpg"))
assert imgs, f"no images in {ds / 'images' / a.split} (run tools/extract_label_frames.py)"
results = {}
for spec in a.model:
    name, det = build(spec)
    det(cv2.imread(str(imgs[0])))  # warm-up
    preds, n_gt, times, cen_hits = [], 0, [], 0
    import torch
    for p in tqdm(imgs, desc=name):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        gt = load_labels(ds / "labels" / a.split / f"{p.stem}.txt", W, H)
        n_gt += len(gt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        d = det(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        d = sorted(d, key=lambda x: -x[4])
        used = set()
        for x0, y0, x1, y1, s in d:
            best, bi = 0, -1
            for i, g in enumerate(gt):
                if i not in used and iou((x0, y0, x1, y1), g) > best:
                    best, bi = iou((x0, y0, x1, y1), g), i
            tp = best >= 0.5
            if tp:
                used.add(bi)
            preds.append((s, tp))
        # centre-distance recall at a moderate threshold
        cd = [x for x in d if x[4] >= 0.25]
        for g in gt:
            gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
            tol = max(8, g[2] - g[0], g[3] - g[1])
            if any(np.hypot((x[0] + x[2]) / 2 - gc[0], (x[1] + x[3]) / 2 - gc[1]) <= tol for x in cd):
                cen_hits += 1
    preds.sort(key=lambda x: -x[0])
    tp = np.cumsum([p[1] for p in preds])
    fp = np.cumsum([not p[1] for p in preds])
    rec = tp / max(1, n_gt)
    prec = tp / np.maximum(1, tp + fp)
    # COCO 101-point interpolated AP
    mprec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else np.array([0])
    ap50 = float(np.mean([mprec[rec >= r].max() if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)]))
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    i = int(np.argmax(f1)) if len(f1) else 0
    results[name] = dict(AP50=round(ap50, 3), P=round(float(prec[i]), 3) if len(prec) else 0,
                         R=round(float(rec[i]), 3) if len(rec) else 0,
                         best_conf=round(float(preds[i][0]), 3) if preds else None,
                         R_centre=round(cen_hits / max(1, n_gt), 3), ms_per_img=round(1000 * float(np.median(times)), 1),
                         n_images=len(imgs), n_labels=n_gt)
    print(name, results[name], flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "eval.json").write_text(json.dumps(results, indent=1))
hdr = ["model", "AP50", "P", "R", "R_centre", "best_conf", "ms_per_img"]
md = [f"Held-out split `{a.split}` of `{ds.name}` ({len(imgs)} images).", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join([k] + [str(v[h]) for h in hdr[1:]]) + " |" for k, v in results.items()]
(out / "eval.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
