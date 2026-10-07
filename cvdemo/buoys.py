"""Buoy detectors (pluggable) + a buoy tracker that works in pan-compensated coordinates.

Detectors return a list of (x0, y0, x1, y1, score) in image coordinates.
  yolo-buoy  : YOLO11 fine-tuned on Grounding-DINO pseudo labels (fast, specialised)
  gdino      : Grounding DINO tiny, zero-shot "buoy." (accurate, slow)
  yoloworld  : YOLO-World v2, open-vocabulary "buoy" (fast, weak on tiny buoys)
  owlv2      : OWLv2 base, open-vocabulary "a buoy" (slow)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"


def _small(b, W, H):
    w, h = b[2] - b[0], b[3] - b[1]
    return w <= 0.06 * W and h <= 0.08 * H and w * h >= 6


def _not_in(b, boxes, thr=0.5):
    for p in boxes:
        x0, y0 = max(b[0], p[0]), max(b[1], p[1])
        x1, y1 = min(b[2], p[2]), min(b[3], p[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        if inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1])) > thr:
            return False
    return True


class YoloBuoy:
    name = "yolo11-buoy (distilled)"

    def __init__(self, weights=None, conf=0.25, imgsz=1088):
        from ultralytics import YOLO
        if weights is None:  # trained weights are committed in weights/, models/ is the local cache
            cands = [MODELS / "buoy_yolo11s.pt", MODELS.parent / "weights" / "buoy_yolo11s.pt"]
            weights = str(next((c for c in cands if c.exists()), cands[0]))
        self.m = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame, persons=()):
        H, W = frame.shape[:2]
        r = self.m.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if _small(b, W, H) and _not_in(b, persons):
                out.append((*b, s))
        return out


class GDinoBuoy:
    name = "grounding-dino-tiny"

    def __init__(self, thr=0.28):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = "IDEA-Research/grounding-dino-tiny"
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(mid)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
        self.thr = thr

    def __call__(self, frame, persons=()):
        from PIL import Image
        H, W = frame.shape[:2]
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=im, text="buoy. person. boat.", return_tensors="pt")
        with self.torch.no_grad():
            o = self.m(**inp)
        r = self.proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(H, W)])[0]
        labels = r.get("text_labels", r["labels"])
        dets = [(l, float(s), [float(v) for v in b]) for l, s, b in zip(labels, r["scores"], r["boxes"])]
        ppl = list(persons) + [b for l, s, b in dets if l == "person" and s > 0.3]
        return [(*b, s) for l, s, b in dets
                if l == "buoy" and s >= self.thr and _small(b, W, H) and _not_in(b, ppl)]


class YoloWorldBuoy:
    name = "yolo-world-v2-m"

    def __init__(self, conf=0.05, imgsz=1280):
        from ultralytics import YOLOWorld
        self.m = YOLOWorld(str(MODELS / "yolov8m-worldv2.pt"))
        self.m.set_classes(["buoy", "person", "boat"])
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame, persons=()):
        H, W = frame.shape[:2]
        r = self.m.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if int(c) == 0 and _small(b, W, H) and _not_in(b, persons):
                out.append((*b, s))
        return out


class Owlv2Buoy:
    name = "owlv2-base"

    def __init__(self, thr=0.12):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        mid = "google/owlv2-base-patch16-ensemble"
        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.m = Owlv2ForObjectDetection.from_pretrained(mid).eval()
        self.thr = thr

    def __call__(self, frame, persons=()):
        from PIL import Image
        H, W = frame.shape[:2]
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.proc(text=[["a buoy", "a person"]], images=im, return_tensors="pt")
        with self.torch.no_grad():
            o = self.m(**inp)
        s = max(W, H)  # OWLv2 pads to a square
        pp = getattr(self.proc, "post_process_grounded_object_detection", None) or self.proc.post_process_object_detection
        r = pp(o, threshold=self.thr, target_sizes=self.torch.tensor([[s, s]]))[0]
        return [(*b, float(sc)) for b, sc, l in zip(r["boxes"].tolist(), r["scores"], r["labels"].tolist())
                if l == 0 and _small(b, W, H) and _not_in(b, persons)]


def build(name: str, **kw):
    return {"yolo-buoy": YoloBuoy, "gdino": GDinoBuoy, "yoloworld": YoloWorldBuoy, "owlv2": Owlv2Buoy}[name](**kw)


# ----------------------------------------------------------------------------------------------
# Tracking

@dataclass
class BuoyTrack:
    tid: int
    hist: list = field(default_factory=list)  # (frame, X, y, w, h, score, observed)  X = pan-compensated x
    misses: int = 0
    hue: list = field(default_factory=list)

    @property
    def last(self):
        return self.hist[-1]

    def n_obs(self):
        return sum(1 for h in self.hist if h[6])

    def predict(self, f):
        """Constant-velocity prediction (pan-compensated coords) from the last observations."""
        obs = [h for h in self.hist if h[6]][-8:]
        if len(obs) < 2:
            return obs[-1][1], obs[-1][2]
        fr = np.array([o[0] for o in obs], float)
        X = np.array([o[1] for o in obs])
        Y = np.array([o[2] for o in obs])
        px, py = np.polyfit(fr, X, 1), np.polyfit(fr, Y, 1)
        return float(np.polyval(px, f)), float(np.polyval(py, f))


class BuoyTracker:
    """Greedy nearest-neighbour tracker. Positions are stored as (X = x + pan, y) so that
    camera panning does not break association; tracks coast through short occlusions (spray)."""

    def __init__(self, max_miss=45, gate=40):
        self.tracks: list[BuoyTrack] = []
        self.dead: list[BuoyTrack] = []
        self.next_id = 1
        self.max_miss, self.gate = max_miss, gate

    def reset(self):
        self.dead += self.tracks
        self.tracks = []

    def update(self, f, dets, pan, frame=None):
        cands = []
        for d in dets:
            x0, y0, x1, y1, s = d
            cands.append(((x0 + x1) / 2 + pan, y1, x1 - x0, y1 - y0, s, d))  # anchor = bottom-centre (waterline)
        preds = [t.predict(f) for t in self.tracks]
        used, pairs = set(), []
        for ti, (px, py) in enumerate(preds):
            for ci, c in enumerate(cands):
                d = np.hypot(c[0] - px, c[1] - py)
                g = self.gate + 2 * max(c[2], c[3])
                if d < g:
                    pairs.append((d, ti, ci))
        pairs.sort()
        tu = set()
        for d, ti, ci in pairs:
            if ti in tu or ci in used:
                continue
            tu.add(ti)
            used.add(ci)
            c = cands[ci]
            t = self.tracks[ti]
            t.hist.append((f, c[0], c[1], c[2], c[3], c[4], True))
            t.misses = 0
            if frame is not None:
                t.hue.append(_hue(frame, c[5]))
        for ti, t in enumerate(self.tracks):
            if ti not in tu:
                t.misses += 1
                px, py = preds[ti]
                l = t.last
                t.hist.append((f, px, py, l[3], l[4], 0.0, False))
        for ci, c in enumerate(cands):
            if ci not in used:
                t = BuoyTrack(self.next_id, [(f, c[0], c[1], c[2], c[3], c[4], True)])
                if frame is not None:
                    t.hue.append(_hue(frame, c[5]))
                self.next_id += 1
                self.tracks.append(t)
        alive = []
        for t in self.tracks:
            lost = t.misses > self.max_miss or (t.misses > 6 and t.n_obs() < 3)
            (self.dead if lost else alive).append(t)
        self.tracks = alive

    def all_tracks(self):
        return self.dead + self.tracks


def _hue(frame, d):
    """Rough colour class of a buoy box: mean hue of its most saturated pixels."""
    x0, y0, x1, y1 = (int(v) for v in d[:4])
    p = cv2.cvtColor(frame[max(0, y0):y1 + 1, max(0, x0):x1 + 1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(p) == 0:
        return -1
    p = p[np.argsort(p[:, 1])[-max(3, len(p) // 4):]]
    return float(np.median(p[:, 0]))
