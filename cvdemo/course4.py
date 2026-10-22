"""Course logic v4: skier movement + buoy proximity, no course centre line, no 3D model.

Idea (from the team):
  * two buoys close to each other at the same depth  -> a GATE: the skier must pass between them
  * a single buoy -> the skier first crosses the middle, then heads to one side: the movement direction
    tells which side the buoy is on (moving left -> left buoy)
  * at the frames where the skier's feet/ski are level with the buoy (same depth = same image row,
    since both are on the water) and close to it, check on which side of the buoy the ski is:
    left buoy -> ski must be LEFT of it (outside), right buoy -> RIGHT of it.
All distances are measured in skier heights (box height), so no camera calibration is needed, and all
comparisons are made inside the same frame, so camera panning does not matter.
"""
from __future__ import annotations

import numpy as np

from . import buoys as B
from . import course as C
from . import geometry as G


def ski_point(r):
    """Where the ski is: a detected ski/board box under the skier if available, else the ankles, else the
    bottom-centre of the skier box. Returns (x, y, skier_height_px)."""
    s = r["skier"]
    h = max(10.0, s[3] - s[1])
    ski = r.get("ski")
    if ski:
        return (ski[0] + ski[2]) / 2, ski[3], h
    k = r.get("kpts")
    if k is not None:
        k = np.asarray(k)
        a = k[[15, 16]]
        a = a[a[:, 2] > 0.3]
        if len(a):
            return float(a[:, 0].mean()), max(float(a[:, 1].max()), s[3] - 0.05 * h), h
    return (s[0] + s[2]) / 2, s[3], h


def _skier_series(frames, s0, s1):
    sf, sx, sy, sh, sX = [], [], [], [], []
    for f in range(s0, s1):
        r = frames[f]
        if r["skier"]:
            x, y, h = ski_point(r)
            sf.append(f)
            sx.append(x)
            sy.append(y)
            sh.append(h)
            sX.append(x + r["pan"])  # pan-compensated: only used for the movement DIRECTION
    return [np.array(v, float) for v in (sf, sx, sy, sh, sX)]


def buoy_events_v4(tracks, frames, s0, s1, fps, motion_s=0.8, near_rows=0.35, near_cols=3.0,
                   gate_gap=2.5, gate_dy=0.25, min_obs_miss=6):
    """near_rows / near_cols / gate_gap / gate_dy are in skier heights. A 'miss' needs a buoy track with at
    least min_obs_miss detections: short tracks next to the ski are mostly the ski / spray, not a buoy."""
    sf, sx, sy, sh, sX = _skier_series(frames, s0, s1)
    if len(sf) < 10:
        return []
    # buoy observations in image coordinates (x, bottom y) per frame
    tobs = []
    for t in tracks:
        hist = [h for h in t.hist if s0 <= h[0] < s1]
        obs = [h for h in hist if h[6]]
        if len(obs) < 3:
            continue
        tobs.append((t, {h[0]: (h[1] - frames[h[0]]["pan"], h[2], h[6]) for h in hist}, len(obs)))
    events = []
    for t, pos, nobs in tobs:
        fr = sorted(pos)
        # frames where buoy and ski are level (same depth) - the buoy recedes up the image past the skier
        cand = []
        for f in fr:
            i = np.searchsorted(sf, f)
            if i >= len(sf) or sf[i] != f:
                continue
            bx, by, observed = pos[f]
            dy = (by - sy[i]) / sh[i]
            cand.append((f, i, bx, by, observed, dy))
        if not cand:
            continue
        # a detection that keeps a fixed position relative to the skier's feet is part of the skier
        # (e.g. a red/pink ski mistaken for a red buoy): a real buoy is anchored and the skier moves past it
        if len(cand) >= 4:
            rx = np.array([(c[2] - sx[c[1]]) / sh[c[1]] for c in cand])
            ry = np.array([(c[3] - sy[c[1]]) / sh[c[1]] for c in cand])
            if np.median(np.abs(rx)) < 1.2 and np.std(rx) < 0.3 and np.std(ry) < 0.2:
                continue
        # crossing = buoy goes from below the ski (closer to camera) to above it
        cross = None
        for (f0, *_, d0), (f1, *_, d1) in zip(cand[:-1], cand[1:]):
            if d0 > 0 >= d1 and f1 - f0 <= fps // 2:
                cross = f1
                break
        if cross is None:
            near = min(cand, key=lambda c: abs(c[5]))
            if abs(near[5]) > near_rows:
                continue
            cross = near[0]
        ci = int(np.searchsorted(sf, cross))
        hpx = sh[ci]
        # ---- gate / centre pair: another buoy at the same depth, close by (in skier heights)
        partner = None
        for t2, pos2, n2 in tobs:
            if t2 is t or cross not in pos2:
                continue
            bx2, by2, _ = pos2[cross]
            bx, by, _ = pos[cross]
            if abs(by2 - by) < gate_dy * hpx and 0.3 * hpx < abs(bx2 - bx) < gate_gap * hpx:
                partner = (t2, bx2)
                break
        bx, by, observed = pos[cross]
        if partner is not None:
            lo, hi = sorted((bx, partner[1]))
            between = lo - 0.2 * hpx <= sx[ci] <= hi + 0.2 * hpx
            events.append(dict(frame=int(cross), t=round(cross / fps, 2), track=t.tid, kind="gate",
                               gate_with=partner[0].tid, through_gate=bool(between), between=bool(between),
                               buoy_x=float(bx), skier_x=float(sx[ci]), n_obs=nobs, predicted=not observed,
                               buoy_X=float(bx + frames[cross]["pan"]), skier_X=float(sX[ci])))
            continue
        # ---- single buoy: side from the skier's movement direction before reaching it
        j0 = int(np.searchsorted(sf, cross - motion_s * fps))
        move = sX[ci] - sX[min(j0, ci)]
        if abs(move) < 0.3 * hpx:  # hardly moving sideways -> use where the buoy is relative to the ski
            move = bx - sx[ci]
        side = 1 if move > 0 else -1
        # judge on the frames where ski and buoy are level and close (median -> robust to single errors)
        # "did the ski get outside the buoy while level with it?": frames within +-0.4 s of the crossing,
        # 80th percentile of the margin (going round the buoy happens at some point; the approach frames,
        # where the skier is still inside, must not outvote it; one noisy frame must not decide it either)
        win = 0.4 * fps
        close = [c for c in cand if abs(c[0] - cross) <= win and abs(c[5]) <= near_rows
                 and abs(c[2] - sx[c[1]]) <= near_cols * sh[c[1]]]
        if not close:
            close = [c for c in cand if c[0] == cross]
        margins = [((sx[c[1]] - c[2]) * side) / sh[c[1]] for c in close]
        m = float(np.percentile(margins, 80))
        seen = any(c[4] for c in close)
        if m > -0.1:
            verdict = "ok"
        elif seen and m < -0.35 and nobs >= min_obs_miss:
            verdict = "miss"
        else:
            verdict = "uncertain"
        events.append(dict(frame=int(cross), t=round(cross / fps, 2), track=t.tid, kind="turn",
                           side="right" if side > 0 else "left", skier_side="left" if side > 0 else "right",
                           outside=m > -0.1, verdict=verdict, margin_h=round(m, 2), margin_px=round(m * hpx, 1),
                           n_close=len(close), buoy_x=float(bx), skier_x=float(sx[ci]), n_obs=nobs,
                           predicted=not seen,
                           # fields used by the shared course fit / renderer
                           buoy_X=float(bx + frames[cross]["pan"]), skier_X=float(sX[ci]), amp=float(hpx * 6)))
    events.sort(key=lambda e: e["frame"])
    # duplicate tracks of one buoy: same kind/side within 0.6 s
    merged = []
    for e in events:
        m = next((k for k in merged if k["kind"] == e["kind"] and k.get("side") == e.get("side")
                  and abs(k["frame"] - e["frame"]) <= 0.6 * fps), None)
        if m is None:
            merged.append(e)
        elif e["n_obs"] > m["n_obs"]:
            merged[merged.index(m)] = e
    return merged


def drop_skier_attached(frames, fps, win_s=0.4, span_s=0.3, tol_x=0.3, tol_y=0.1):
    """Remove 'buoy' detections that are part of the skier (a red/pink ski, a glove, spray at the ski):
    near the ski point and staying at about the same position relative to it for >= span_s (detections within
    +-win_s at the same relative offset). A real buoy is anchored, so it slides past the moving skier.
    Detections are sparse (the ski is found in some frames only), so neighbours are searched in a window, not
    at fixed frame offsets. Mutates frames[f]['buoys']; returns how many were removed."""
    w = max(1, int(win_s * fps))
    rel = {}
    for f, r in enumerate(frames):
        if r["skier"] and r["buoys"]:
            x, y, h = ski_point(r)
            rel[f] = [(((b[0] + b[2]) / 2 - x) / h, (b[3] - y) / h) for b in r["buoys"]]
    removed = 0
    for f, rs in rel.items():
        keep = []
        for b, (rx, ry) in zip(frames[f]["buoys"], rs):
            attached = False
            if abs(rx) < 1.5 and -0.2 < ry < 0.5:
                same = [g for g in range(f - w, f + w + 1) if g != f and any(
                    abs(rx2 - rx) < tol_x and abs(ry2 - ry) < tol_y for rx2, ry2 in rel.get(g, []))]
                attached = bool(same) and max(same + [f]) - min(same + [f]) >= span_s * fps
            if attached:
                removed += 1
            else:
                keep.append(b)
        frames[f]["buoys"] = keep
    return removed


def analyse_course_v4(frames, meta, boat_view=None):
    fps, shots = meta["fps"], meta["shots"]
    bounds = list(zip(shots, shots[1:] + [len(frames)]))
    if boat_view is None:
        boat_view = {s0: C.shot_is_boat_view(frames, s0, s1, meta["width"], meta["height"]) for s0, s1 in bounds}
    n_attached = drop_skier_attached(frames, fps)
    events, tracks_all, ends = [], [], {}
    for s0, s1 in bounds:
        if not boat_view[s0]:
            continue
        tr = B.BuoyTracker(max_miss=int(2 * fps), frame_size=(meta["width"], meta["height"]))
        for f in range(s0, s1):
            if frames[f]["buoys"] is not None:
                tr.update(f, frames[f]["buoys"], frames[f]["pan"])
        tracks = tr.all_tracks()
        tracks_all += [(s0, s1, t) for t in tracks]
        events += buoy_events_v4(tracks, frames, s0, s1, fps)
        ends[s0] = G.run_end_frame(frames, s0, s1, fps)
    turns_all = [e for e in events if e["kind"] == "turn"]
    passes = G.split_passes(turns_all, fps)
    for p in passes:
        G.check_alternation(p)

    def end_for(p):
        return ends[max(s for s in ends if s <= p[0]["frame"])]

    courses = [G.fit_course(p, fps, end_frame=end_for(p)) for p in passes]
    # only the ENTRY / EXIT gates have to be passed through; pairs met in the middle of a pass are the boat
    # guide buoys (the boat goes between them, the skier swings across) -> shown but not judged
    for g in (e for e in events if e["kind"] == "gate"):
        inside = any(p[0]["frame"] < g["frame"] < p[-1]["frame"] for p in passes)
        g["role"] = "guide" if inside else ("entry" if any(g["frame"] <= p[0]["frame"] for p in passes) else "exit")
        if g["role"] == "guide":
            g["through_gate"] = None
    scores = [G.course_score(c) for c in courses]
    for c in courses:
        for sl in c.get("slots", []):
            if sl["event"] is not None:
                sl["event"]["slot"] = sl["slot"]
    main = max(range(len(passes)), key=lambda i: (scores[i]["seen"], len(passes[i])), default=None)
    fitted = {id(sl["event"]) for c in courses for sl in c.get("slots", []) if sl["event"] is not None}
    shown = [e for e in events if e["kind"] != "turn" or id(e) in fitted]
    return dict(events=events, shown_events=shown, tracks=tracks_all, passes=passes, courses=courses,
                scores=scores, main=main, boat_view=boat_view, skier_attached_removed=n_attached,
                main_score=scores[main] if main is not None else dict(estimate=0, confirmed=0, seen=0))
