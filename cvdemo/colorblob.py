"""Buoy candidates from colour contrast, no model: a small spot whose colour differs strongly from a large,
nearly uniform ring of water around it ("centre-surround").

Per pixel, in the perceptual Lab colour space:
  centre   = mean colour in a small window (about one buoy)
  surround = mean colour in a large window (several buoys wide)
  contrast = |centre - surround| (Lab distance; ~ how different a colour the human eye sees)
  uniform  = the surround's own colour spread is small (water, not spray / shore / the skier)
Spots with high contrast on a uniform surround, roughly round and buoy-sized, are candidates. White buoys
on blue-green water are found too (brightness is part of Lab), white spray is not: spray is not compact and
its surround is not uniform.
"""
from __future__ import annotations

import cv2
import numpy as np


def _box_mean(x, k):
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


class ColorBlobBuoy:
    name = "colour centre-surround (no model)"

    def __init__(self, sizes=(7, 13, 21), min_contrast=20.0, max_surround_std=20.0, min_score=0.0, verifier=None):
        self.sizes, self.min_contrast, self.max_surround_std = sizes, min_contrast, max_surround_std
        self.min_score = min_score
        self.skier_box = None
        self.overlay = None  # set per video: static_overlay_mask(video)
        self.verifier = verifier  # optional: re-scores candidate crops (e.g. ClipVerifier)
        self.max_verify = 15

    def zone(self, H):
        s = self.skier_box
        if s is not None and s[3] - s[1] > 20:
            h = s[3] - s[1]
            return int(max(0, s[3] - 1.0 * h)), int(min(H, s[3] + 4.0 * h))
        return int(0.45 * H), H

    def candidates(self, img):
        """img: BGR band. Returns [(x0, y0, x1, y1, score, contrast, surround_std)]."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255  # L to 0..100 like a and b spread
        lab[..., 1:] -= 128
        best = np.zeros(img.shape[:2], np.float32)
        best_std = np.zeros_like(best)
        for c in self.sizes:
            s = 5 * c
            cen = np.stack([_box_mean(lab[..., i], c) for i in range(3)], -1)
            sur = np.stack([_box_mean(lab[..., i], s) for i in range(3)], -1)
            sq = np.stack([_box_mean(lab[..., i] ** 2, s) for i in range(3)], -1)
            std = np.sqrt(np.clip(sq - sur ** 2, 0, None).sum(-1))
            con = np.sqrt(((cen - sur) ** 2).sum(-1))
            take = con > best
            best[take], best_std[take] = con[take], std[take]
        mask = ((best > self.min_contrast) & (best_std < self.max_surround_std)).astype(np.uint8)
        # pixel-level: the spot itself must differ from the local surround, not just be near a big edge
        px = np.sqrt(((lab - np.stack([_box_mean(lab[..., i], 5 * self.sizes[1]) for i in range(3)], -1)) ** 2)
                     .sum(-1))
        mask &= (px > self.min_contrast).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab_, st, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 8 or w > 70 or h > 70:
                continue
            fill = area / float(w * h)
            if not (0.4 < w / max(h, 1) < 2.5) or fill < 0.45:  # roughly round / compact
                continue
            sel = lab_[y:y + h, x:x + w] == i
            con = float(best[y:y + h, x:x + w][sel].mean())
            sstd = float(best_std[y:y + h, x:x + w][sel].mean())
            score = min(1.0, con / 60.0) * (1 - min(1.0, sstd / self.max_surround_std)) ** 0.5 * fill ** 0.5
            if score >= self.min_score:
                out.append((float(x), float(y), float(x + w), float(y + h), round(score, 3), round(con, 1),
                            round(sstd, 1)))
        return out

    def __call__(self, frame, persons=()):
        H, W = frame.shape[:2]
        y0, y1 = self.zone(H)
        out = []
        for b in self.candidates(frame[y0:y1]):
            bb = (b[0], b[1] + y0, b[2], b[3] + y0, b[4])
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            if any(p[0] <= cx <= p[2] and p[1] <= cy <= p[1] + 0.65 * (p[3] - p[1]) for p in persons):
                continue  # on the skier's body (feet / ski area kept)
            if self.overlay is not None and self.overlay[int(min(H - 1, cy)), int(min(W - 1, cx))]:
                continue  # TV graphics
            out.append(bb)
        if self.verifier is not None and out:
            out = sorted(out, key=lambda b: -b[4])[:self.max_verify]  # strongest colour contrast first
            ps = self.verifier(frame, out)
            out = [(*b[:4], round(float(p), 3)) for b, p in zip(out, ps)]
        return out


def static_overlay_mask(video, n=40, std_thr=6.0, dilate=9):
    """TV graphics (scoreboards, logos, captions) stay fixed on screen while the camera moves: pixels whose
    brightness hardly changes across the video are overlay. Returns a uint8 mask (1 = overlay)."""
    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = []
    for f in np.linspace(0, max(0, N - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            acc.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(acc) < 5:
        return None
    a = np.stack(acc)
    # captions come and go: a pixel is overlay if it is static in a large share of frame pairs
    static = (np.abs(np.diff(a, axis=0)) < std_thr).mean(0) > 0.35
    m = static.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
