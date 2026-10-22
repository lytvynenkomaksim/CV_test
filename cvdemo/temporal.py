"""Buoy detection over time, at full resolution: a buoy is something that stays at the same place on the
water for several frames; sparkle and spray flicker.

Per frame:
  1. predict every known buoy's position: previous position + camera pan + its own drift in the image
     (a buoy slides through the picture as the boat moves; the drift is learned per buoy);
  2. local search: look for it only in a small window around the prediction (template match against its
     last appearance + colour-contrast check) - cheap, so it runs on every frame at full resolution;
  3. full detection (YOLOE 'buoy' + colour centre-surround with CLIP check) in the zone near the skier every
     `global_every` frames, to find new buoys; detections next to a known buoy feed that buoy.
After the clip (offline, the video is recorded): a track is a buoy if it was seen in >= `min_hits` frames
and its best evidence was strong enough; then it counts in ALL its frames, also the ones before it was
confirmed, and short gaps are filled by interpolation. Everything else is dropped.
"""
from __future__ import annotations

import cv2
import numpy as np


def stands_out(img, b, min_contrast=14.0):
    """Lab colour distance between the box interior and a ring of water around it (2.2x the box). A buoy
    stands out; a patch of plain water / shore texture that a template happened to match does not."""
    w, h = b[2] - b[0], b[3] - b[1]
    if w < 2 or h < 2:
        return False
    H, W = img.shape[:2]
    X0, Y0 = int(max(0, b[0] - 0.6 * w)), int(max(0, b[1] - 0.6 * h))
    X1, Y1 = int(min(W, b[2] + 0.6 * w)), int(min(H, b[3] + 0.6 * h))
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return False
    lab = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = np.zeros(lab.shape[:2], bool)
    ix0, iy0 = int(b[0] - X0 + 0.2 * w), int(b[1] - Y0 + 0.2 * h)
    ix1, iy1 = int(b[2] - X0 - 0.2 * w), int(b[3] - Y0 - 0.2 * h)
    inner[max(0, iy0):max(iy0 + 1, iy1), max(0, ix0):max(ix0 + 1, ix1)] = True
    ring = np.ones_like(inner)
    ring[max(0, int(b[1] - Y0)):int(b[3] - Y0), max(0, int(b[0] - X0)):int(b[2] - X0)] = False
    if inner.sum() < 2 or ring.sum() < 4:
        return False
    d = lab[inner].mean(0) - lab[ring].mean(0)
    d[0] *= 100 / 255
    return float(np.sqrt((d ** 2).sum())) >= min_contrast


class Track:
    _next = 0

    def __init__(self, f, box, score, img):
        self.id = Track._next
        Track._next += 1
        self.boxes = {f: list(box)}      # frame -> [x0, y0, x1, y1] (image coordinates)
        self.how = {f: "det"}            # det | local
        self.best = score                # best detector / CLIP evidence
        self.scores = [score]
        self.v = np.zeros(2)             # own drift in the image, px / frame (pan removed)
        self.miss = 0
        self.tpl = self._patch(img, box)

    @staticmethod
    def _patch(img, b, pad=0.6):
        w, h = b[2] - b[0], b[3] - b[1]
        x0, y0 = int(max(0, b[0] - pad * w)), int(max(0, b[1] - pad * h))
        x1, y1 = int(min(img.shape[1], b[2] + pad * w)), int(min(img.shape[0], b[3] + pad * h))
        p = img[y0:y1, x0:x1]
        return p.copy() if p.size else None

    def last(self):
        f = max(self.boxes)
        return f, self.boxes[f]


class TemporalBuoys:
    def __init__(self, detectors, global_every=2, min_hits=4, min_best=0.35, search=2.5, max_miss=12,
                 ncc_min=0.55):
        """detectors: list of callables (frame, persons) -> [(x0, y0, x1, y1, score)], each with optional
        .skier_box attribute (zone). Scores should be comparable (0..1)."""
        self.dets, self.global_every = detectors, global_every
        self.min_hits, self.min_best, self.search, self.max_miss, self.ncc_min = (
            min_hits, min_best, search, max_miss, ncc_min)
        self.tracks, self.dead = [], []
        self.prev_pan = None

    # ------------------------------------------------------------------ per frame
    def update(self, f, img, pan, skier_box=None, persons=()):
        dpan = 0.0 if self.prev_pan is None else pan - self.prev_pan
        self.prev_pan = pan
        H, W = img.shape[:2]
        live = []
        for t in self.tracks:
            lf, b = t.last()
            k = f - lf
            dx, dy = -dpan * k + t.v[0] * k, t.v[1] * k
            pb = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
            hit = self._local(img, t, pb)
            if hit is not None:
                nb, ncc = hit
                c_old = np.array([(b[0] + b[2]) / 2 - dpan * k, (b[1] + b[3]) / 2])  # pan removed
                c_new = np.array([(nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2])
                t.v = 0.7 * t.v + 0.3 * (c_new - c_old) / max(1, k)
                t.boxes[f], t.how[f], t.miss = nb, "local", 0
                if ncc > 0.8:
                    t.tpl = Track._patch(img, nb)
            else:
                t.miss += 1
            gone = pb[2] < 0 or pb[0] > W or pb[3] < 0 or pb[1] > H
            (self.dead if (t.miss > self.max_miss or gone) else live).append(t)
        self.tracks = live
        if f % self.global_every == 0:
            for d in self.dets:
                if hasattr(d, "skier_box"):
                    d.skier_box = skier_box
                for b in d(img, persons):
                    self._associate(f, img, list(b[:4]), float(b[4]))

    def _local(self, img, t, pb):
        if t.tpl is None or t.tpl.size == 0:
            return None
        w, h = pb[2] - pb[0], pb[3] - pb[1]
        r = self.search * max(w, h, 8)
        x0, y0 = int(max(0, pb[0] - r)), int(max(0, pb[1] - r))
        x1, y1 = int(min(img.shape[1], pb[2] + r)), int(min(img.shape[0], pb[3] + r))
        win = img[y0:y1, x0:x1]
        th, tw = t.tpl.shape[:2]
        if win.shape[0] <= th or win.shape[1] <= tw:
            return None
        m = cv2.matchTemplate(win, t.tpl, cv2.TM_CCOEFF_NORMED)
        _, ncc, _, loc = cv2.minMaxLoc(m)
        if ncc < self.ncc_min:
            return None
        padx, pady = (tw - w) / 2, (th - h) / 2
        nx, ny = x0 + loc[0] + padx, y0 + loc[1] + pady
        nb = [nx, ny, nx + w, ny + h]
        if not stands_out(img, nb):  # matched plain water / shore texture, not the buoy
            return None
        return nb, ncc

    def _associate(self, f, img, b, s):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best, bd = None, 1e9
        for t in self.tracks:
            lf, tb = t.last()
            if f - lf > 2:
                continue
            tx, ty = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            d = np.hypot(cx - tx, cy - ty)
            if d < max(10, 1.2 * max(tb[2] - tb[0], tb[3] - tb[1])) and d < bd:
                best, bd = t, d
        if best is not None:
            best.boxes[f], best.how[f], best.miss = b, "det", 0
            best.best = max(best.best, s)
            best.scores.append(s)
            best.tpl = Track._patch(img, b)
        else:
            self.tracks.append(Track(f, b, s, img))

    # ------------------------------------------------------------------ offline result
    def result(self, max_gap=6):
        """frame -> [(x0, y0, x1, y1, score, track_id, how)] for confirmed buoy tracks (gaps interpolated)."""
        out = {}
        for t in self.tracks + self.dead:
            n = len(t.boxes)
            n_det = sum(1 for h in t.how.values() if h == "det")
            if n < self.min_hits or t.best < self.min_best or n_det < 2:
                continue
            fs = sorted(t.boxes)
            conf = min(1.0, 0.5 * t.best + 0.5 * float(np.mean(sorted(t.scores)[-3:])))
            for a, b in zip(fs, fs[1:] + [None]):
                out.setdefault(a, []).append((*t.boxes[a], conf, t.id, t.how[a]))
                if b is not None and 1 < b - a <= max_gap:
                    for g in range(a + 1, b):
                        w = (g - a) / (b - a)
                        bb = [(1 - w) * p + w * q for p, q in zip(t.boxes[a], t.boxes[b])]
                        out.setdefault(g, []).append((*bb, conf, t.id, "gap"))
        return out


class ScaledDetector:
    """Stretch a detector's scores (YOLOE gives 0.05-0.6 for real buoys) so evidence is comparable."""

    def __init__(self, d, k):
        self.d, self.k = d, k

    @property
    def skier_box(self):
        return self.d.skier_box

    @skier_box.setter
    def skier_box(self, v):
        self.d.skier_box = v

    def __call__(self, frame, persons=()):
        return [(*b[:4], min(1.0, self.k * b[4])) for b in self.d(frame, persons)]


class VideoBuoys:
    """Pipeline adapter: full-resolution multi-frame buoy detection for one video (feed every frame, then
    finish() returns frame -> buoys). Untrained: YOLOE 'buoy' + colour centre-surround with CLIP check."""
    name = "multi-frame: YOLOE 'buoy' + colour/CLIP, local search, >=4 frames (no training, full res)"

    def __init__(self, video, global_every=2):
        from . import buoys as B
        from .colorblob import ColorBlobBuoy, static_overlay_mask
        from .verify import ClipVerifier
        self.cb = ColorBlobBuoy(verifier=ClipVerifier())
        self.cb.overlay = static_overlay_mask(video)
        self.tb = TemporalBuoys([ScaledDetector(B.YoloEZoneBuoy(conf=0.03, imgsz=1920), 3.0), self.cb],
                                global_every=global_every)

    def feed(self, f, frame, pan, skier_box, persons):
        self.tb.update(f, frame, pan, skier_box=skier_box, persons=persons)

    def finish(self):
        return self.tb.result()
