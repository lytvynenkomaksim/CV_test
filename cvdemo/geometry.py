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


class FlowPanEstimator:
    """Horizontal camera pan from sparse optical flow on the far background (shore / trees).

    Same idea as the camera-motion compensation (GMC) in BoT-SORT: track corner features with
    pyramidal Lucas-Kanade between consecutive frames, then fit a similarity transform with RANSAC.
    Water, the skier and broadcast overlays are masked out, so moving spray/wakes do not
    drag the estimate (the phase-correlation version drifted by >1000 px over a pass).
    """

    def __init__(self, band=(0.04, 0.50), scale=0.5, max_pts=400):
        self.band, self.scale, self.max_pts = band, scale, max_pts
        self.prev = None
        self.pan = 0.0
        self.last_inliers = 0

    def reset(self):
        self.prev = None

    def _mask(self, shape, exclude):
        h, w = shape
        m = np.zeros((h, w), np.uint8)
        m[int(h * self.band[0]):int(h * self.band[1]), :] = 255
        # broadcast overlays live in the corners (score bug, logos, name banner)
        m[:int(h * 0.16), :int(w * 0.22)] = 0
        m[:int(h * 0.16), int(w * 0.78):] = 0
        for b in exclude:
            x0, y0, x1, y1 = (int(v * self.scale) for v in b[:4])
            pw, ph = (x1 - x0) // 2 + 4, (y1 - y0) // 2 + 4
            m[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw] = 0
        return m

    def update(self, frame, exclude=()):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        if self.prev is not None and self.prev.shape == g.shape:
            pts = cv2.goodFeaturesToTrack(self.prev, self.max_pts, 0.01, 6, mask=self._mask(g.shape, exclude))
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    if M is not None and inl is not None and inl.sum() >= 6:
                        self.last_inliers = int(inl.sum())
                        dx = M[0, 2] + (M[0, 0] - 1) * g.shape[1] / 2  # translation at image centre
                        self.pan -= dx / self.scale
        self.prev = g
        return self.pan


class PanEstimator:
    """Horizontal camera pan from the far background band using phase correlation.
    (First version; drifts on water/spray. Kept for comparison, see FlowPanEstimator.)"""

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


def buoy_events(tracks, frames, seg_start, seg_end, turn_ratio=0.45):
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
        # trajectory = observations + the tracker's projections while the buoy was hidden
        # (spray / wave), so a crossing can be judged even if the buoy is covered at that moment
        traj = [h for h in t.hist if seg_start <= h[0] < seg_end and h[0] >= obs[0][0]]
        hf = [h[0] for h in traj]
        hX = [h[1] for h in traj]
        hY = [h[2] for h in traj]
        hidden = {h[0] for h in traj if not h[6]}
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
        predicted = cross in hidden or cross > obs[-1][0]
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


def buoy_events_v2(tracks, frames, seg_start, seg_end, fps, window_s=2.5, turn_ratio=0.45, dedup_s=0.6):
    """Buoy crossings judged *locally* (v2).

    v1 used one course-centre line per shot in pan-compensated coordinates; with an operator panning
    up to ~90 deg through each turn, the accumulated pan (and the small-angle x+pan approximation)
    drifts by more than a frame width, so most buoys ended up on the wrong side.
    v2:
      * centre/amplitude from the skier's swing in a +-window_s window around the crossing only
        (one window covers the previous and the next turn, i.e. both extremes of the swing);
      * inside/outside from image coordinates *at the crossing frame* (pan cancels exactly);
      * duplicate tracks of the same buoy (lost and re-found) are merged.
    """
    W = int(window_s * fps)
    sf = [f for f in range(seg_start, seg_end) if frames[f] and frames[f]["skier_X"] is not None]
    if len(sf) < 10:
        return []
    sX = np.array([frames[f]["skier_X"] for f in sf])
    sY = [frames[f]["foot_y"] for f in sf]
    events = []
    for t in tracks:
        obs = [h for h in t.hist if h[6] and seg_start <= h[0] < seg_end]
        if len(obs) < 3:
            continue
        traj = [h for h in t.hist if seg_start <= h[0] < seg_end and h[0] >= obs[0][0]]
        hf = [h[0] for h in traj]
        hX = [h[1] for h in traj]
        hY = [h[2] for h in traj]
        hidden = {h[0] for h in traj if not h[6]}
        prev, cross = None, None
        for f in range(hf[0], hf[-1] + 1):
            by, fy = _interp(hf, hY, f), _interp(sf, sY, f)
            if by is None or fy is None:
                continue
            d = by - fy
            if prev is not None and prev > 0 >= d:
                cross = f
                break
            prev = d
        if cross is None:
            continue
        win = (np.array(sf) >= cross - W) & (np.array(sf) <= cross + W)
        if win.sum() < 10:
            continue
        lo, hi = np.percentile(sX[win], 5), np.percentile(sX[win], 95)
        C, A = (lo + hi) / 2, max(10.0, (hi - lo) / 2)
        pan = frames[cross]["pan"] if frames[cross] else 0.0
        bX = _interp(hf, hX, cross)
        sXc = _interp(sf, list(sX), cross)
        off = bX - C
        r = off / A
        w = float(np.median([h[3] for h in obs]))
        ev = dict(frame=cross, t=round(cross / fps, 2), track=t.tid, buoy_x=round(bX - pan, 1),
                  skier_x=round(sXc - pan, 1), buoy_X=bX, skier_X=sXc, centre=C, amp=A, ratio=round(r, 2),
                  predicted=bool(cross in hidden or cross > obs[-1][0]), n_obs=len(obs),
                  hue=round(float(np.median(t.hue)), 1) if t.hue else -1)
        if abs(r) >= turn_ratio:
            side = 1 if off > 0 else -1
            margin = (sXc - bX) * side  # same frame -> identical to image-coordinate difference
            sk = frames[cross]["skier"] if frames[cross] else None
            sw = (sk[2] - sk[0]) if sk else 40.0
            outside = bool(margin > -0.5 * w)
            # a miss is only called when the buoy was really seen and the skier is clearly inside;
            # close calls and positions projected through spray/fast pans are "uncertain"
            if outside:
                verdict = "ok"
            elif ev["predicted"] or margin > -0.6 * sw:
                verdict = "uncertain"
            else:
                verdict = "miss"
            ev.update(kind="turn", side="right" if side > 0 else "left",
                      skier_side="left" if side > 0 else "right",
                      outside=outside, verdict=verdict, margin_px=round(margin, 1))
        else:
            ev.update(kind="centre")
        events.append(ev)
    events.sort(key=lambda e: e["frame"])
    # gates: two buoys reached at (almost) the same moment, one on each side of the skier.
    # (entry/exit gate; the skier must pass BETWEEN them, so they are not turn buoys)
    used = set()
    for i, e1 in enumerate(events):
        for j in range(i + 1, len(events)):
            e2 = events[j]
            if (e2["frame"] - e1["frame"]) > 0.35 * fps:
                break
            if i in used or j in used:
                continue
            l, r = sorted((e1, e2), key=lambda e: e["buoy_x"])
            gap = r["buoy_x"] - l["buoy_x"]
            amp = (e1["amp"] + e2["amp"]) / 2
            mid_X = (l["buoy_X"] + r["buoy_X"]) / 2
            # gate buoys are ~2.3 m apart and centred on the boat path; turn buoys are ~23 m apart
            if gap < 25 or gap > 0.8 * amp or abs(mid_X - (e1["centre"] + e2["centre"]) / 2) > 0.5 * amp:
                continue
            between = l["buoy_x"] - 0.25 * gap <= (e1["skier_x"] + e2["skier_x"]) / 2 <= r["buoy_x"] + 0.25 * gap
            for e in (e1, e2):
                e.update(kind="gate", gate_with=(e2 if e is e1 else e1)["track"], through_gate=bool(between))
                e.pop("side", None)
            used |= {i, j}
    # merge duplicates: same kind/side, close in time and in position
    merged = []
    for e in events:
        m = next((k for k in merged if k["kind"] == e["kind"] and k.get("side") == e.get("side")
                  and abs(k["frame"] - e["frame"]) <= dedup_s * fps
                  and abs(k["buoy_X"] - e["buoy_X"]) < max(80.0, 0.35 * e["amp"])), None)
        if m is None:
            merged.append(e)
        elif e["n_obs"] > m["n_obs"]:
            e["merged"] = m.get("merged", []) + [m["track"]]
            merged[merged.index(m)] = e
        else:
            m.setdefault("merged", []).append(e["track"])
    return merged


def split_passes(events, fps, gap_s=6.0):
    """A video can contain several passes (or a replay): split turn events on long gaps."""
    turns = [e for e in events if e["kind"] == "turn"]
    passes, cur = [], []
    for e in turns:
        if cur and (e["frame"] - cur[-1]["frame"]) / fps > gap_s:
            passes.append(cur)
            cur = []
        cur.append(e)
    if cur:
        passes.append(cur)
    return passes


def check_alternation(turns):
    """Turn buoys alternate sides. Flag consecutive turns on the same side (one of them is
    probably mis-sided or a false buoy) so they can be reviewed / excluded from the score."""
    for a, b in zip(turns[:-1], turns[1:]):
        if a["side"] == b["side"]:
            b["alternation_ok"] = False
    return turns


def run_end_frame(frames, s0, s1, fps, lost_s=2.5):
    """Last frame of the run inside a shot: the skier is lost for > lost_s (fall / let go)
    or the shot ends. Slots after it are 'not reached'."""
    last, gap = None, 0
    for f in range(s0, s1):
        if frames[f]["skier"]:
            last, gap = f, 0
        else:
            gap += 1
            if last is not None and gap > lost_s * fps:
                return last
    return last if last is not None else s0


def fit_course(turns, fps, n_buoys=6, period_range=(1.8, 3.4), default_period=2.5, end_frame=None):
    """Fit the slalom course rhythm to the detected turn-buoy crossings of one pass.

    The six turn buoys are passed at a near-constant interval T (~2.3-2.8 s) and alternate sides.
    We estimate T, lay a grid of slots t0 + k*T, keep the 6 consecutive slots that explain most
    events, keep the best event per slot, fix the alternation parity by (n_obs-weighted) majority and
    drop events that contradict it (duplicates of other buoys, far buoys, mis-sided tracks).
    Returns a list of 6 slot dicts: {k, t, side, event or None, status: ok/miss/unseen/conflict}.
    """
    if not turns:
        return []
    t = np.array([e["frame"] / fps for e in turns])
    gaps = np.diff(np.sort(t))
    cand = [g for g in gaps if period_range[0] <= g <= period_range[1]]
    cand += [g / 2 for g in gaps if 2 * period_range[0] <= g <= 2 * period_range[1]]
    T = float(np.median(cand)) if cand else default_period
    tol = 0.3 * T
    # anchor: the grid phase that explains most events
    best = None
    for t0 in t:
        k = np.round((t - t0) / T)
        fit = np.abs(t - (t0 + k * T)) <= tol
        if best is None or fit.sum() > best[0]:
            best = (fit.sum(), t0)
    t0 = best[1]
    ks = np.round((t - t0) / T).astype(int)
    on_grid = np.abs(t - (t0 + ks * T)) <= tol
    # the 6 consecutive slots holding most events
    kmin = min(ks[on_grid]) if on_grid.any() else 0
    kmax = max(ks[on_grid]) if on_grid.any() else 0
    starts = range(kmin - n_buoys + 1, kmax + 1)
    # ties -> latest window start, i.e. the first detected turn becomes buoy 1 (clips start before the course)
    k0 = max(starts, key=lambda s0: (sum(1 for k, g in zip(ks, on_grid) if g and s0 <= k < s0 + n_buoys), s0))
    # alternation parity by weighted majority
    vote = 0.0
    for e, k, g in zip(turns, ks, on_grid):
        if g and k0 <= k < k0 + n_buoys:
            sgn = 1 if e["side"] == "right" else -1
            vote += sgn * (-1) ** (k - k0) * (1 + 0.1 * e["n_obs"])
    s0 = 1 if vote >= 0 else -1
    slots = []
    for i in range(n_buoys):
        k = k0 + i
        side = "right" if s0 * (-1) ** i > 0 else "left"
        cands = [e for e, kk, g in zip(turns, ks, on_grid) if g and kk == k]
        agree = [e for e in cands if e["side"] == side]
        conflict = [e for e in cands if e["side"] != side]
        ev = max(agree, key=lambda e: (not e["predicted"], e["n_obs"])) if agree else None
        status = "unseen" if ev is None else ev.get("verdict", "ok" if ev["outside"] else "miss")
        if ev is None and conflict:
            status = "conflict"
        if ev is None and end_frame is not None and (t0 + k * T) * fps > end_frame + 0.3 * T * fps:
            status = "not_reached"
        slots.append(dict(slot=i + 1, t=round(t0 + k * T, 2), side=side, status=status,
                          event=ev, rejected=[e["track"] for e in conflict]))
    return dict(period_s=round(T, 2), slots=slots)


def course_score(course):
    """Score from fitted slots. 'confirmed' counts consecutive buoys seen rounded outside;
    'estimate' also lets unseen buoys pass (the skier was not seen failing them)."""
    confirmed = estimate = 0
    stop_c = stop_e = False
    for s in course.get("slots", []):
        if s["status"] in ("miss", "not_reached"):
            stop_c = stop_e = True
        if not stop_e:
            estimate += 1
        if s["status"] != "ok":  # unseen / uncertain / conflict: cannot be confirmed
            stop_c = True
        if not stop_c:
            confirmed += 1
    seen = sum(s["status"] in ("ok", "miss") for s in course.get("slots", []))
    return dict(estimate=estimate, confirmed=confirmed, seen=seen)


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
