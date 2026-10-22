"""Tow-rope detection: the long thin straight line from the skier's hands down towards the boat pylon.

Classical (no model): edges + probabilistic Hough in the region below the hands; keep long segments whose
extension passes through the hands, pick the best-supported direction. Returns the rope angle in the image
(0 deg = straight down, + = rope leans to the right going down) and the point where it leaves the frame,
which over a run converges on the pylon position (fixed for a boat-mounted camera).
"""
from __future__ import annotations

import cv2
import numpy as np


def hands(r, min_conf=0.3):
    k = r.get("kpts")
    if k is None:
        return None
    k = np.asarray(k)
    w = k[[9, 10]]
    w = w[w[:, 2] > min_conf]
    if not len(w):
        return None
    return float(w[:, 0].mean()), float(w[:, 1].mean())


def detect_rope(img, r, min_len_h=1.2, max_dist_h=0.25):
    """img: BGR frame, r: per-frame record with 'skier' box and 'kpts'. Returns dict or None."""
    if not r.get("skier"):
        return None
    hnd = hands(r)
    s = r["skier"]
    h = max(10.0, s[3] - s[1])
    if hnd is None:
        hnd = ((s[0] + s[2]) / 2, s[1] + 0.45 * h)
    H, W = img.shape[:2]
    y0 = int(max(0, hnd[1]))
    roi = img[y0:H]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    segs = cv2.HoughLinesP(e, 1, np.pi / 360, threshold=40, minLineLength=int(min_len_h * h * 0.5),
                           maxLineGap=int(0.15 * h))
    if segs is None:
        return None
    hx, hy = hnd[0], hnd[1] - y0
    best = {}
    for x1, y1, x2, y2 in segs.reshape(-1, 4).tolist():
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dy, dx = y2 - y1, x2 - x1
        if dy < 0.3 * abs(dx) + 1:  # the rope goes DOWN from the hands (not the horizon / wake edges)
            continue
        # distance of the hands from the segment's infinite line
        d = abs(dx * (hy - y1) - dy * (hx - x1)) / np.hypot(dx, dy)
        if d > max_dist_h * h or y1 > hy + 3 * h:
            continue
        ang = np.degrees(np.arctan2(dx, dy))
        key = int(round(ang / 2))  # 2-degree bins: collinear pieces of the same rope vote together
        best.setdefault(key, []).append((np.hypot(dx, dy), ang, (x1, y1 + y0, x2, y2 + y0)))
    if not best:
        return None
    key = max(best, key=lambda k_: sum(v[0] for v in best[k_]))
    tot = sum(v[0] for v in best[key])
    if tot < min_len_h * h:
        return None
    ang = float(np.average([v[1] for v in best[key]], weights=[v[0] for v in best[key]]))
    t = np.tan(np.radians(ang))
    exit_x = hnd[0] + t * (H - hnd[1])  # where the rope line meets the bottom edge
    return dict(angle_deg=round(ang, 1), support_px=round(tot), hands=hnd, exit_x=float(exit_x),
                segments=[v[2] for v in best[key]])
