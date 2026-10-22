"""Ski detection without any model: the long, thin object at the skier's feet.

Tried first: classical filters (colour masks, straight-edge / Hough lines at the feet). They mostly picked
the shins, the rope or spray lines, so the ski is found with the same untrained open-vocabulary model as
the buoys (YOLOE, prompt "water ski"), on a small crop around the ankles. Returns
[x0, y0, x1, y1, end1, end2, angle_deg, score], or None (then the ankles are used instead).
"""
from __future__ import annotations

import cv2
import numpy as np


def feet(rec):
    s = rec["skier"]
    h = max(10.0, s[3] - s[1])
    k = rec.get("kpts")
    if k is not None:
        k = np.asarray(k)
        a = k[[15, 16]]
        a = a[a[:, 2] > 0.3]
        if len(a):
            return float(a[:, 0].mean()), float(a[:, 1].max()), h
    return (s[0] + s[2]) / 2, s[3] - 0.05 * h, h


class SkiDetector:
    """UNTRAINED: YOLOE-26L segmentation with the prompt "water ski" on a crop around the feet.
    The mask is cleaned: the part above the ankles (legs) is cut off, rope-thin shapes are rejected, and the
    shape must touch the feet. The ski = minimum-area rectangle of the remaining mask."""
    name = "ski: YOLOE-26L 'water ski' at the feet (no training)"

    def __init__(self, weights="yoloe-26l-seg.pt", conf=0.15, min_len_h=0.3):
        from pathlib import Path
        from ultralytics import YOLOE
        self.m = YOLOE(str(Path(__file__).resolve().parent.parent / "models" / weights))
        P = ["water ski", "ski"]
        self.m.set_classes(P, self.m.get_text_pe(P))
        self.conf, self.min_len_h = conf, min_len_h

    def __call__(self, frame, rec):
        if not rec.get("skier"):
            return None
        fx, fy, h = feet(rec)
        H, W = frame.shape[:2]
        x0, x1 = int(max(0, fx - 1.0 * h)), int(min(W, fx + 1.0 * h))
        y0, y1 = int(max(0, fy - 0.45 * h)), int(min(H, fy + 0.45 * h))
        if x1 - x0 < 16 or y1 - y0 < 8:
            return None
        crop = frame[y0:y1, x0:x1]
        r = self.m.predict(crop, imgsz=640, conf=self.conf, verbose=False)[0]
        if r.masks is None:
            return None
        fxl, fyl = fx - x0, fy - y0
        best = None
        for k, c in enumerate(r.boxes.conf.tolist()):
            mk = cv2.resize(r.masks.data[k].cpu().numpy().astype(np.uint8), (x1 - x0, y1 - y0),
                            interpolation=cv2.INTER_NEAREST)
            mk[: max(0, int(fyl - 0.12 * h))] = 0  # cut the legs off
            ys, xs = np.nonzero(mk)
            if len(xs) < 10 or np.min(np.hypot(xs - fxl, ys - fyl)) > 0.15 * h:
                continue
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            L, Wd = max(rw, rh), min(rw, rh)
            if L < self.min_len_h * h or Wd < 0.03 * h or L / max(Wd, 1) < 2.0:  # too short / rope-thin / blob
                continue
            if best is None or c > best[0]:
                best = (c, (cx, cy), (rw, rh), ang, L)
        if best is None:
            return None
        c, (cx, cy), (rw, rh), ang, L = best
        a = np.radians(ang if rw >= rh else ang + 90)
        d = np.array([np.cos(a), np.sin(a)]) * L / 2
        e1, e2 = np.array([cx + x0, cy + y0]) - d, np.array([cx + x0, cy + y0]) + d
        return [float(min(e1[0], e2[0])), float(min(e1[1], e2[1])), float(max(e1[0], e2[0])),
                float(max(e1[1], e2[1])), [float(e1[0]), float(e1[1])], [float(e2[0]), float(e2[1])],
                round(float(np.degrees(a)) % 180, 1), round(float(c), 2)]
