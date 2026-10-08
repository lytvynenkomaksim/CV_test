"""Course logic v3: everything in METRES on the water (boat frame), using the calibrated camera.

Per boat-camera shot:
  1. calibrate the camera (cvdemo.camera.ShotCalibration) -> pixel <-> (X, Z) metres;
     X = sideways from the course centre line (the boat path, X = 0), Z = distance behind the boat
  2. track buoys in metres: an anchored buoy keeps its X and recedes at boat speed (Z += v dt), which
     also gives an exact position while it is hidden by spray;
     only buoys the skier has NOT passed yet are tracked: a track is closed once the buoy is more than
     2.5 m behind the skier, and nothing is started behind the skier
  3. classify: |X| small and paired with a buoy on the other side at the same Z -> gate / boat-guide pair;
     |X| large and alone -> turn buoy
  4. crossing = the moment buoy and skier are at the same Z; turn buoy judged by sideways margin in metres
     (skier outside the buoy line?), gate judged by the skier being between the pair
  5. course fit: each buoy's position along the lake s = v*t - Z is constant; turn buoys are 41 m apart and
     alternate sides, the gates are 27 m before buoy 1 / after buoy 6 -> slots 1..6, and the expected position
     of every turn buoy (also ones never detected) can be projected back into the image
If a shot cannot be calibrated (e.g. zoomed/operated camera) the v2 image-space logic is used instead.
"""
from __future__ import annotations

import numpy as np

from . import buoys as B
from . import camera as CAM
from . import course as C
from . import geometry as G

TURN_SPACING_M = 41.0
GATE_TO_FIRST_M = 27.0
TURN_OFFSET_M = 11.5


def skier_foot(r):
    """Where the ski is on the water: ankles from the skeleton if visible, else the box bottom-centre."""
    k = r.get("kpts")
    if k is not None:
        k = np.asarray(k)
        a = k[[15, 16]]
        a = a[a[:, 2] > 0.3]
        if len(a):
            return float(a[:, 0].mean()), float(a[:, 1].max())
    s = r["skier"]
    return (s[0] + s[2]) / 2, s[3]


class MetricTrack:
    def __init__(self, tid, f, X, Z, s):
        self.tid = tid
        self.obs = [(f, X, Z, s)]  # observed (frame, X, Z, score)
        self.closed = None  # reason
        self.misses = 0

    def predict(self, f, v, fps):
        fs = np.array([o[0] for o in self.obs[-12:]], float)
        Xs = np.array([o[1] for o in self.obs[-12:]])
        Zs = np.array([o[2] for o in self.obs[-12:]]) - v * fs / fps  # -> along-course position (constant)
        return float(np.median(Xs)), float(np.median(Zs) + v * f / fps)

    @property
    def X(self):
        return float(np.median([o[1] for o in self.obs]))


def metric_tracks(frames, cal, s0, s1, fps):
    v = cal.v
    tracks, active, nid = [], [], 1
    hist = {}  # frame -> list of (tid, X, Z, observed)
    for f in range(s0, s1):
        r = frames[f]
        zs = None
        if r["skier"]:
            u, vv = skier_foot(r)
            xs_, zs_ = cal.to_water(f, u, vv)
            if np.isfinite(xs_):
                zs = float(zs_)
        dets = []
        for b in (r["buoys"] or []):
            X, Z = cal.to_water(f, (b[0] + b[2]) / 2, b[3])
            if np.isfinite(X) and 1.0 < Z < 60:
                dets.append((float(X), float(Z), b[4]))
        preds = [t.predict(f, v, fps) for t in active]
        used, pairs = set(), []
        for ti, (px, pz) in enumerate(preds):
            for di, (X, Z, s) in enumerate(dets):
                d = np.hypot(X - px, (Z - pz) * 0.5)  # depth is less precise than sideways position
                if d < 1.5 + 0.08 * pz:
                    pairs.append((d, ti, di))
        tu = set()
        for d, ti, di in sorted(pairs):
            if ti in tu or di in used:
                continue
            tu.add(ti)
            used.add(di)
            active[ti].obs.append((f, *dets[di]))
            active[ti].misses = 0
        for ti, t in enumerate(active):
            if ti not in tu:
                t.misses += 1
        for di, (X, Z, s) in enumerate(dets):
            if di in used:
                continue
            if zs is not None and Z > zs + 1.0:
                continue  # already behind the skier: passed, not interesting
            t = MetricTrack(nid, f, X, Z, s)
            nid += 1
            active.append(t)
        still = []
        for ti, t in enumerate(active):
            px, pz = t.predict(f, v, fps)
            if zs is not None and pz > zs + 2.5:
                t.closed = "passed"
            elif t.misses > 2 * fps:
                t.closed = "lost"
            elif t.misses > 6 and len(t.obs) < 3:
                t.closed = "noise"
            if t.closed:
                tracks.append(t)
            else:
                still.append(t)
                hist.setdefault(f, []).append((t.tid, px, pz, t.misses == 0, t.misses))
        active = still
    tracks += active
    return [t for t in tracks if len(t.obs) >= 3], hist


def prepare_video(video, frames, meta, boat_model=None, progress=True):
    """One pass over the video: camera pan (for pixel tracking) + rotation increments (for the camera model),
    and which shots are the boat camera. Mutates frames[f]['pan'] / ['skier_X']."""
    import cv2
    from tqdm import tqdm
    shots = meta["shots"]
    cap = cv2.VideoCapture(str(video))
    pe, cm, inc = G.FlowPanEstimator(), CAM.CameraMotion(), []
    for f in tqdm(range(len(frames)), desc="camera motion", leave=False, disable=not progress):
        ok, img = cap.read()
        if not ok:
            break
        if f in shots:
            pe.reset()
            cm.reset()
        ex = [frames[f]["skier"]] if frames[f]["skier"] else []
        frames[f]["pan"] = pe.update(img, ex)
        inc.append(cm.update(img, ex))
        if frames[f]["skier"]:
            frames[f]["skier_X"] = (frames[f]["skier"][0] + frames[f]["skier"][2]) / 2 + frames[f]["pan"]
    inc += [(0.0, 0.0, 0.0)] * (len(frames) - len(inc))
    boat_view = {}
    for s0, s1 in zip(shots, shots[1:] + [len(frames)]):
        if boat_model is None:
            boat_view[s0] = C.shot_is_boat_view(frames, s0, s1, meta["width"], meta["height"])
            continue
        votes = []
        for f in np.linspace(s0, s1 - 1, min(5, s1 - s0)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                bx = boat_model.predict(img, classes=[8], conf=0.35, verbose=False)[0].boxes.xyxy.tolist()
                votes.append(any((b[2] - b[0]) * (b[3] - b[1]) > 0.01 * meta["width"] * meta["height"]
                                 and b[3] < 0.97 * meta["height"] for b in bx))
        boat_view[s0] = not (votes and np.mean(votes) >= 0.5)
    cap.release()
    return inc, boat_view


def calibration_inputs(frames, meta, s0, s1):
    """Skier foot points and pixel buoy tracks of one shot (inputs of ShotCalibration.fit)."""
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    tr = B.BuoyTracker(max_miss=int(2 * fps), frame_size=(W, H))
    for f in range(s0, s1):
        if frames[f]["buoys"] is not None:
            tr.update(f, frames[f]["buoys"], frames[f]["pan"])
    ptracks = [[(h[0], h[1] - frames[h[0]]["pan"], h[2]) for h in t.hist if h[6]]
               for t in tr.all_tracks() if t.n_obs() >= 4]
    skier_obs = [(f, *skier_foot(frames[f])) for f in range(s0, s1) if frames[f]["skier"]]
    return skier_obs, ptracks


def is_women(name):
    return any(k in name.lower() for k in ("women", "girls", "julie", "alexandra", "allie", "regina"))


def analyse_course_metric(frames, meta, inc, name, boat_view=None, rig=None):
    fps, shots = meta["fps"], meta["shots"]
    W, H = meta["width"], meta["height"]
    bounds = list(zip(shots, shots[1:] + [len(frames)]))
    if boat_view is None:
        boat_view = {s0: C.shot_is_boat_view(frames, s0, s1, W, H) for s0, s1 in bounds}
    rope = CAM.rope_length_from_name(name)
    women = is_women(name)
    out = dict(shots=[], events=[], courses=[], fallback_shots=[])
    for s0, s1 in bounds:
        if not boat_view[s0] or s1 - s0 < 3 * fps:
            continue
        skier_obs, ptracks = calibration_inputs(frames, meta, s0, s1)
        cal = CAM.ShotCalibration(s0, s1, W, H, fps, inc)
        cal.fit(skier_obs, ptracks, rope, v_prior=15.3 if women else 16.1, fixed=rig)
        shot = dict(start=s0, end=s1, calibration=cal.report, metric=cal.ok)
        out["shots"].append(shot)
        if not cal.ok:
            out["fallback_shots"].append(s0)
            continue
        shot["cal"] = cal
        tracks, hist = metric_tracks(frames, cal, s0, s1, fps)
        shot["hist"] = hist
        # skier trajectory in metres
        sk = {}
        for f in range(s0, s1):
            if frames[f]["skier"]:
                X, Z = cal.to_water(f, *skier_foot(frames[f]))
                if np.isfinite(X):
                    sk[f] = (float(X), float(Z))
        shot["skier"] = sk
        sf = np.array(sorted(sk))
        if len(sf) < 10:
            continue
        sX = np.array([sk[f][0] for f in sf])
        sZ = np.array([sk[f][1] for f in sf])
        # classify tracks + crossings
        info = []
        for t in tracks:
            fs = np.array([o[0] for o in t.obs])
            s_course = np.median([cal.v * o[0] / fps - o[2] for o in t.obs])  # position along the lake
            info.append(dict(track=t.tid, X=t.X, s=float(s_course), first=int(fs[0]), last=int(fs[-1]),
                             n_obs=len(t.obs), closed=t.closed))
        for a in info:  # pairs: two buoys at the same course position on opposite sides, close to centre
            a["kind"] = "turn" if abs(a["X"]) >= 4.5 else ("centre" if abs(a["X"]) < 3.0 else "unknown")
        for i, a in enumerate(info):
            for b in info[i + 1:]:
                if a["kind"] == b["kind"] == "centre" and abs(a["s"] - b["s"]) < 4 and a["X"] * b["X"] < 0:
                    a["pair"], b["pair"] = b["track"], a["track"]
        events = []
        for a in info:
            if a["kind"] == "unknown":
                continue
            # crossing: the buoy (Z = v t - s) reaches the skier's Z
            zb = cal.v * sf / fps - a["s"]
            d = zb - sZ
            idx = np.where((d[:-1] < 0) & (d[1:] >= 0) & (np.diff(sf) <= fps // 2))[0]
            if not len(idx):
                continue
            i = idx[0]
            fcross = int(sf[i + 1])
            xs = float(np.interp(fcross, sf, sX))
            seen = a["first"] <= fcross <= a["last"] + 3
            ev = dict(frame=fcross, t=round(fcross / fps, 2), track=a["track"], kind=a["kind"], buoy_X=a["X"],
                      skier_X=xs, s=a["s"], predicted=not seen, n_obs=a["n_obs"])
            if a["kind"] == "turn":
                side = 1 if a["X"] > 0 else -1
                margin = xs * side - abs(a["X"])
                verdict = "ok" if margin > -0.3 else ("miss" if (seen and margin < -1.0) else "uncertain")
                ev.update(side="right" if side > 0 else "left", margin_m=round(margin, 2),
                          outside=margin > -0.3, verdict=verdict)
            else:
                ev.update(between=abs(xs) < max(abs(a["X"]) + 0.5, 1.5))
            events.append(ev)
        events.sort(key=lambda e: e["frame"])
        # merge duplicates of the same physical buoy (same course position and side)
        merged = []
        for e in events:
            m = next((k for k in merged if k["kind"] == e["kind"] and abs(k["s"] - e["s"]) < 4
                      and np.sign(k["buoy_X"]) == np.sign(e["buoy_X"])), None)
            if m is None:
                merged.append(e)
            elif e["n_obs"] > m["n_obs"]:
                merged[merged.index(m)] = e
        events = merged
        # gates: centre pairs before the first / after the last turn buoy
        turns = [e for e in events if e["kind"] == "turn"]
        cents = [e for e in events if e["kind"] == "centre"]
        paired = {a["track"] for a in info if "pair" in a}
        if turns:
            for e in cents:  # a gate is a centred PAIR before the first / after the last turn buoy
                if e["track"] in paired and (e["s"] < turns[0]["s"] - 10 or e["s"] > turns[-1]["s"] + 10):
                    e["kind"] = "gate"
        course = fit_metric_course(turns, cal, fps)
        out["events"] += events
        out["courses"].append(course)
        shot["course"] = course
    return out


def fit_metric_course(turns, cal, fps, n=6):
    """Assign turn buoys to slots 1..6 by their position along the lake (41 m apart, alternating sides)."""
    if not turns:
        return dict(slots=[], s1=None)
    s = np.array([e["s"] for e in turns])
    # spacing is 41 m on the lake; the measured value absorbs any remaining scale error of the camera model
    best = None
    for D in np.arange(25.0, 50.5, 1.0):
        for anchor in s:  # try each detected turn buoy as buoy k, keep the grid explaining most buoys
            k = np.round((s - anchor) / D)
            fit = np.abs(s - (anchor + k * D)) < 0.18 * D
            lo = int(k[fit].min())
            for first in range(lo - n + 1, lo + 1):
                inside = fit & (k >= first) & (k < first + n)
                # most buoys; then alternation consistency; then closest to the nominal 41 m
                alt = sum(1 for e, kk, ok in zip(turns, k, inside) if ok) and _alternation(turns, k, inside)
                score = (inside.sum(), alt, -abs(D - TURN_SPACING_M * 0.8), first)
                if best is None or score > best[0]:
                    best = (score, anchor + first * D, D)
    s1, D = best[1], best[2]
    vote = sum((1 if e["side"] == "right" else -1) * (-1) ** int(round((e["s"] - s1) / D))
               * (1 + 0.1 * e["n_obs"]) for e in turns)
    side1 = 1 if vote >= 0 else -1
    slots = []
    for i in range(n):
        sc = s1 + i * D
        side = "right" if side1 * (-1) ** i > 0 else "left"
        cands = [e for e in turns if abs(e["s"] - sc) < 0.18 * D and e["side"] == side]
        ev = max(cands, key=lambda e: (not e["predicted"], e["n_obs"])) if cands else None
        if ev is not None:
            ev["slot"] = i + 1
        slots.append(dict(slot=i + 1, s=round(float(sc), 1), side=side, event=ev,
                          status="unseen" if ev is None else ev["verdict"]))
    offs = [abs(e["buoy_X"]) for e in turns]
    return dict(s1=float(s1), side1=side1, spacing_m=float(D), slots=slots,
                turn_offset_m=float(np.median(offs)) if offs else TURN_OFFSET_M)


def _alternation(turns, k, inside):
    """How many detected turn buoys agree with alternating sides along the grid (best parity)."""
    sg = [(1 if e["side"] == "right" else -1) * (-1) ** int(kk) for e, kk, ok in zip(turns, k, inside) if ok]
    return max(sum(1 for x in sg if x > 0), sum(1 for x in sg if x < 0))


def course_score(course):
    return G.course_score(course)


# --------------------------------------------------------------------------------------------- rendering
def _dash(img, pts, col, th=1, on=10, off=8):
    import cv2
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = np.hypot(b[0] - a[0], b[1] - a[1])
        if acc % (on + off) < on:
            cv2.line(img, a, b, col, th, cv2.LINE_AA)
        acc += seg


def render_metric(video, out_path, frames, meta, ana, labels, note=""):
    import subprocess

    import cv2

    from . import pose as P
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    cap = cv2.VideoCapture(str(video))
    tmp = str(out_path) + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    shot_of = {}
    for sh in ana["shots"]:
        for f in range(sh["start"], sh["end"]):
            shot_of[f] = sh
    evs = ana["events"]
    MW, MH = int(0.2 * W), int(0.36 * H)  # mini-map size
    trail = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= len(frames):
            break
        r = frames[f]
        sh = shot_of.get(f)
        cal = sh.get("cal") if sh else None
        mm = None
        if cal is not None:
            course = sh.get("course") or {}
            off = course.get("turn_offset_m", TURN_OFFSET_M)
            zs = sh["skier"].get(f, (None, None))[1]
            # course centre line (boat path) and the two buoy lines
            Zl = np.linspace(2.0, 60.0, 60)
            for Xc, col, th in ((0.0, (255, 255, 255), 2), (off, (0, 140, 255), 1), (-off, (0, 140, 255), 1)):
                u, v = cal.to_pixel(f, np.full_like(Zl, Xc), Zl)
                pts = [(int(a), int(b)) for a, b in zip(u, v) if np.isfinite(a) and -W < a < 2 * W and 0 <= b < H]
                if len(pts) > 1:
                    (_dash(img, pts, col, th) if Xc else cv2.polylines(img, [np.array(pts)], False, col, th, cv2.LINE_AA))
            if zs is not None:
                u, v = cal.to_pixel(f, np.array([0.0]), np.array([zs]))
                if np.isfinite(u[0]) and 0 <= u[0] < W:
                    cv2.putText(img, "course centre", (int(u[0]) + 5, int(v[0]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
            # tracked buoys (observed / hidden), only those not yet passed by the skier
            mm = dict(buoys=[], expected=[], skier=sh["skier"].get(f), cal=cal, off=off)
            tracked_s = []
            for tid, X, Z, obs, misses in sh["hist"].get(f, []):
                u, v = cal.to_pixel(f, np.array([X]), np.array([Z]))
                tracked_s.append(cal.v * f / fps - Z)
                mm["buoys"].append((X, Z, obs))
                if not np.isfinite(u[0]):
                    continue
                rad = int(max(5, 0.35 * cal.f / max(Z, 1)))
                col = (0, 165, 255) if abs(X) >= 4.5 else (0, 255, 255)
                if obs:
                    cv2.circle(img, (int(u[0]), int(v[0]) - rad // 2), rad + 3, col, 2, cv2.LINE_AA)
                else:
                    r2 = rad + 3 + int(0.4 * misses)
                    for k in range(0, 360, 40):
                        cv2.ellipse(img, (int(u[0]), int(v[0]) - rad // 2), (r2, r2), 0, k, k + 20, col, 1, cv2.LINE_AA)
                    cv2.putText(img, "hidden (est.)", (int(u[0]) - r2, int(v[0]) - r2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, col, 1, cv2.LINE_AA)
            # expected turn buoys from the fitted course (also ones never detected), not yet passed
            for sl in course.get("slots", []):
                Z = cal.v * f / fps - sl["s"]
                if zs is None or not (1.5 < Z < zs + 1.0) or any(abs(sl["s"] - ts) < 5 for ts in tracked_s):
                    continue
                X = off if sl["side"] == "right" else -off
                mm["expected"].append((X, Z, sl["slot"]))
                u, v = cal.to_pixel(f, np.array([X]), np.array([Z]))
                if np.isfinite(u[0]) and 0 <= u[0] < W and 0 <= v[0] < H:
                    cv2.drawMarker(img, (int(u[0]), int(v[0])), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 12, 1)
                    cv2.putText(img, f"buoy {sl['slot']} (expected)", (int(u[0]) + 6, int(v[0]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        # skier + skeleton
        if r["skier"]:
            x0, y0, x1, y1 = r["skier"][:4]
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1)
            if r["kpts"] is not None:
                P.draw_skeleton(img, np.array(r["kpts"]))
        # event banners
        for e in evs:
            if 0 <= f - e["frame"] < int(1.2 * fps):
                if e["kind"] == "turn" and not e.get("slot"):
                    continue  # not part of the fitted course (duplicate / far buoy): no banner
                if e["kind"] == "turn":
                    col = {"ok": (0, 200, 0), "miss": (0, 0, 255), "uncertain": (0, 200, 255)}[e["verdict"]]
                    slot = f"buoy {e['slot']} " if e.get("slot") else ""
                    msg = {"ok": "OUTSIDE - OK", "miss": "INSIDE - MISS", "uncertain": "too close to call"}[e["verdict"]]
                    msg = f"{slot}{e['side'].upper()}: skier {msg} ({e['margin_m']:+.1f} m)"
                    if e["predicted"]:
                        msg += " [buoy hidden]"
                elif e["kind"] == "gate":
                    col = (0, 200, 0) if e["between"] else (0, 0, 255)
                    msg = "GATE: between the gate buoys" if e["between"] else "GATE: missed"
                else:
                    continue
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (W // 2 - tw // 2 - 8, 60), (W // 2 + tw // 2 + 8, 92), (0, 0, 0), -1)
                cv2.putText(img, msg, (W // 2 - tw // 2, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        # mini-map (top-down, boat at the top)
        if mm is not None:
            x0m, y0m = W - MW - 10, 110
            ov = img.copy()
            cv2.rectangle(ov, (x0m, y0m), (x0m + MW, y0m + MH), (40, 40, 40), -1)
            img = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
            sx, sz = MW / 30.0, MH / 34.0

            def P2(X, Z):
                return int(x0m + MW / 2 + X * sx), int(y0m + 8 + Z * sz)
            cv2.line(img, P2(0, 0), P2(0, 33), (255, 255, 255), 1)
            for Xo in (mm["off"], -mm["off"]):
                _dash(img, [P2(Xo, z) for z in np.linspace(0, 33, 12)], (0, 140, 255), 1, 4, 4)
            cv2.drawMarker(img, P2(0, 0), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)
            for X, Z, obs in mm["buoys"]:
                if 0 <= Z <= 33:
                    col = (0, 165, 255) if abs(X) >= 4.5 else (0, 255, 255)
                    cv2.circle(img, P2(X, Z), 4, col, -1 if obs else 1, cv2.LINE_AA)
            for X, Z, k in mm["expected"]:
                if 0 <= Z <= 33:
                    cv2.drawMarker(img, P2(X, Z), (200, 200, 200), cv2.MARKER_TILTED_CROSS, 7, 1)
            if mm["skier"]:
                X, Z = mm["skier"]
                trail.append((X, Z - cal.v / fps * 0))
                cv2.line(img, P2(0, 0), P2(X, Z), (180, 180, 180), 1, cv2.LINE_AA)  # rope
                cv2.circle(img, P2(X, Z), 5, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "top view (m)", (x0m + 4, y0m + MH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # HUD
        sh_c = (sh or {}).get("course") or {}
        done = [sl for sl in sh_c.get("slots", []) if sl["event"] is not None and sl["event"]["frame"] <= f]
        sym = dict(ok="OK", miss="X", uncertain="?")
        lines = [f"models: {labels.get('skier', '')} | pose {labels.get('pose', '')} | buoys {labels.get('buoy', '')}",
                 "buoys: " + " ".join(f"{sl['slot']}{sl['side'][0].upper()}:{sym[sl['status']]}" for sl in done)]
        if cal is not None:
            rep = sh["calibration"]
            lines.append(f"camera model: f={rep['f_px']:.0f}px h={rep['cam_height_m']}m boat {rep['boat_speed_kmh']} km/h"
                         + (f" | {note}" if note else ""))
        elif sh is not None or f in shot_of:
            lines.append("camera not calibrated for this shot")
        y = H - 12 - 18 * (len(lines) - 1)
        ov = img.copy()
        cv2.rectangle(ov, (0, y - 16), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        for i, l in enumerate(lines):
            cv2.putText(img, l, (8, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        f += 1
    vw.release()
    cap.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], check=True)
    import os
    os.remove(tmp)
