"""Camera motion, scene cuts, course centre line and buoy-pass (event) logic.

Coordinate convention
---------------------
The camera sits on the boat and looks backwards; the operator pans to follow the skier.
We estimate the horizontal pan between consecutive frames from the far background
(shoreline / trees) with phase correlation and accumulate it, so that

    X = x_image + pan_t

is a pan-compensated horizontal coordinate. For a level camera mounted on the boat axis,
the boat's straight path projects to a *vertical line* X = C in these coordinates, so the
course centre line is a single number per uninterrupted shot.

Buoy pass rule (what the demo checks)
-------------------------------------
Buoys and skier both float on the water plane, so when the receding buoy reaches the skier's
depth their image rows (waterline y) coincide. At that crossing moment:
  * buoy far from the centre line (|X_b - C| large)  -> TURN buoy. The skier must be OUTSIDE it:
        buoy right of centre -> skier must be right of the buoy,  left -> left.
  * buoy close to the centre line                    -> CENTRE buoy (gate / boat guide buoy),
        reported but not judged.
"""
from __future__ import annotations

import cv2
import numpy as np


class PanEstimator:
    """Horizontal camera pan from the far background band using phase correlation."""

    def __init__(self, band=(0.12, 0.45), cols=(0.12, 0.88), scale=0.5):
        self.band, self.cols, self.scale = band, cols, scale
        self.prev = None
        self.win = None
        self.pan = 0.0

    def _prep(self, frame):
        H, W = frame.shape[:2]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = g[int(H * self.band[0]):int(H * self.band[1]), int(W * self.cols[0]):int(W * self.cols[1])]
        g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        g = np.float32(g)
        g -= cv2.GaussianBlur(g, (0, 0), 8)  # remove illumination gradients
        return g

    def reset(self):
        self.prev = None

    def update(self, frame):
        g = self._prep(frame)
        if self.prev is not None and self.prev.shape == g.shape:
            if self.win is None or self.win.shape != g.shape[::-1]:
                self.win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
            if resp > 0.05 and abs(dx) < g.shape[1] * 0.3:
                self.pan -= dx / self.scale
        self.prev = g
        return self.pan


class CutDetector:
    """Hard scene cut (broadcast edit / replay) via HSV histogram correlation."""

    def __init__(self, thr=0.55):
        self.prev, self.thr = None, thr

    def __call__(self, frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        cut = self.prev is not None and cv2.compareHist(self.prev, h, cv2.HISTCMP_CORREL) < self.thr
        self.prev = h
        return cut


def segment_centre(skier_X):
    """Centre line and swing amplitude of the skier in pan-compensated coords."""
    X = np.asarray([x for x in skier_X if x is not None], float)
    if len(X) < 10:
        return None, None
    lo, hi = np.percentile(X, 8), np.percentile(X, 92)
    return (lo + hi) / 2, max(10.0, (hi - lo) / 2)


def _interp(hist_f, vals, f):
    if f < hist_f[0] or f > hist_f[-1]:
        return None
    return float(np.interp(f, hist_f, vals))


def buoy_events(tracks, frames, seg_start, seg_end, turn_ratio=0.45, max_extrap=20):
    """Find skier/buoy crossings inside one shot.

    frames: list indexed by frame number of dicts with keys 'skier_X', 'foot_y', 'pan'.
    Returns list of event dicts sorted by frame.
    """
    sk = [frames[f]["skier_X"] if frames[f] else None for f in range(seg_start, seg_end)]
    C, A = segment_centre(sk)
    if C is None:
        return [], None, None
    sf = [f for f in range(seg_start, seg_end) if frames[f] and frames[f]["skier_X"] is not None]
    sX = [frames[f]["skier_X"] for f in sf]
    sY = [frames[f]["foot_y"] for f in sf]
    events = []
    for t in tracks:
        obs = [h for h in t.hist if h[6] and seg_start <= h[0] < seg_end]
        if len(obs) < 3:
            continue
        # extend observed trajectory by a short constant-velocity extrapolation (buoy hidden by spray)
        hf = [h[0] for h in obs]
        hX = [h[1] for h in obs]
        hY = [h[2] for h in obs]
        if len(obs) >= 4:
            k = min(10, len(obs))
            vx = np.polyfit(hf[-k:], hX[-k:], 1)
            vy = np.polyfit(hf[-k:], hY[-k:], 1)
            for f in range(hf[-1] + 1, min(seg_end, hf[-1] + max_extrap)):
                hf.append(f)
                hX.append(float(np.polyval(vx, f)))
                hY.append(float(np.polyval(vy, f)))
        prev = None
        cross = None
        for f in range(hf[0], hf[-1] + 1):
            by = _interp(hf, hY, f)
            fy = _interp(sf, sY, f)
            if by is None or fy is None:
                continue
            d = by - fy  # >0 buoy closer to camera than skier
            if prev is not None and prev[1] > 0 >= d:
                cross = f
                break
            prev = (f, d)
        if cross is None:
            continue
        bX, sXc = _interp(hf, hX, cross), _interp(sf, sX, cross)
        off = bX - C
        r = off / A
        predicted = cross > obs[-1][0]
        w = np.median([h[3] for h in obs])
        hue = float(np.median(t.hue)) if t.hue else -1
        ev = dict(frame=cross, track=t.tid, buoy_X=bX, skier_X=sXc, centre=C, amp=A, ratio=round(r, 2),
                  predicted=bool(predicted), hue=round(hue, 1), n_obs=len(obs))
        if abs(r) >= turn_ratio:
            side = 1 if off > 0 else -1
            margin = (sXc - bX) * side
            ev.update(kind="turn", side="right" if side > 0 else "left",
                      skier_side="left" if side > 0 else "right",  # skier faces the camera -> mirrored
                      outside=bool(margin > -0.5 * w), margin_px=round(margin, 1))
        else:
            ev.update(kind="centre")
        events.append(ev)
    events.sort(key=lambda e: e["frame"])
    return events, C, A


def score_pass(events):
    """Simplified slalom score: consecutive turn buoys rounded on the outside (max 6)."""
    turns = [e for e in events if e["kind"] == "turn"]
    score = 0
    for e in turns:
        if not e["outside"]:
            break
        score += 1
    return dict(turn_buoys_seen=len(turns), turn_buoys_outside=sum(e["outside"] for e in turns),
                score=min(score, 6))
