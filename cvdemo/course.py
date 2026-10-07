"""Course-level reasoning shared by the pipeline and tools/rescore.py (v2).

frames (per-frame dicts with pan / skier / buoys / skier_X / foot_y [/ boats]) ->
  boat-camera shots -> buoy tracks -> local pass judgements -> gates -> passes -> 6-slot course fit
"""
from __future__ import annotations

import numpy as np

from . import buoys as B
from . import geometry as G


def shot_is_boat_view(frames, s0, s1, W, H):
    """A shot filmed from another camera (shore / judge tower / replay) shows the towing boat itself.
    Uses the COCO 'boat' boxes recorded per frame; our own stern touches the bottom edge, so it is ignored."""
    votes = []
    for f in range(s0, s1, max(1, (s1 - s0) // 15)):
        boats = frames[f].get("boats")
        if boats is None:
            return True  # not recorded (older runs) -> assume boat view
        votes.append(any((b[2] - b[0]) * (b[3] - b[1]) > 0.01 * W * H and b[3] < 0.97 * H for b in boats))
    return not (votes and np.mean(votes) >= 0.5)


def analyse_course(frames, meta, boat_view=None):
    fps, shots = meta["fps"], meta["shots"]
    bounds = list(zip(shots, shots[1:] + [len(frames)]))
    if boat_view is None:
        boat_view = {s0: shot_is_boat_view(frames, s0, s1, meta["width"], meta["height"]) for s0, s1 in bounds}
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
        events += G.buoy_events_v2(tracks, frames, s0, s1, fps)
        ends[s0] = G.run_end_frame(frames, s0, s1, fps)
    passes = G.split_passes(events, fps)
    for p in passes:
        G.check_alternation(p)

    def end_for(p):
        return ends[max(s for s in ends if s <= p[0]["frame"])]

    courses = [G.fit_course(p, fps, end_frame=end_for(p)) for p in passes]
    scores = [G.course_score(c) for c in courses]
    for c in courses:
        for sl in c.get("slots", []):
            if sl["event"] is not None:
                sl["event"]["slot"] = sl["slot"]
    main = max(range(len(passes)), key=lambda i: (scores[i]["seen"], len(passes[i])), default=None)
    # events shown in videos: course-fitted turns + gates (+ centre buoys)
    fitted = {id(sl["event"]) for c in courses for sl in c.get("slots", []) if sl["event"] is not None}
    shown = [e for e in events if e["kind"] != "turn" or id(e) in fitted]
    return dict(events=events, shown_events=shown, tracks=tracks_all, passes=passes, courses=courses,
                scores=scores, main=main, boat_view=boat_view,
                main_score=scores[main] if main is not None else dict(estimate=0, confirmed=0, seen=0))


SYM = dict(ok="+", miss="x", unseen="?", conflict="!", uncertain="~", not_reached="-")


def slots_str(course):
    return " ".join(f"{s['side'][0].upper()}{SYM[s['status']]}" for s in course.get("slots", []))
