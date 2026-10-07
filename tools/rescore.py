"""Re-run the course geometry / buoy-pass logic on saved per-frame detections.

The pipeline stores every frame's skier box, skeleton and buoy detections in
<run>/<video>.frames.json, so improving the *reasoning* never needs the GPU again:

    python tools/rescore.py --run results/colab_t4/demo --out results/rescore_v2 [--render]

Steps per video: recompute camera pan with the optical-flow estimator (needs the video file,
CPU only, ~5-20 ms/frame) -> buoy tracking -> local-window pass judgement (v2) -> passes/score.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import buoys as B  # noqa: E402
from cvdemo import course as C  # noqa: E402
from cvdemo import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--frames-dir", help="where *.frames.json live (default: --run)")
ap.add_argument("--out", default=str(ROOT / "results/rescore_v2"))
ap.add_argument("--only", nargs="*")
ap.add_argument("--render", action="store_true", help="also render annotated videos")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fdir = Path(a.frames_dir or a.run)

from ultralytics import YOLO  # noqa: E402

BOAT = YOLO(str(ROOT / "models/yolo11s.pt"))  # COCO 'boat' -> external camera shot detector
rows = []
jsons = sorted(p for p in Path(a.run).glob("*.json") if not p.name.endswith(".frames.json") and p.stem != "summary")
for js in tqdm(jsons, desc="videos"):
    if a.only and not any(o in js.stem for o in a.only):
        continue
    res = json.loads(js.read_text())
    meta = res["meta"]
    frames = json.loads((fdir / f"{js.stem}.frames.json").read_text())
    video = ROOT / meta["video"] if not Path(meta["video"]).is_absolute() else Path(meta["video"])
    if not video.exists():  # Colab paths -> local repo paths
        video = ROOT / meta["video"].split("CV_test/")[-1]
    fps, shots = meta["fps"], meta["shots"]
    # 1) camera pan (optical flow on the background), reset at scene cuts
    cap = cv2.VideoCapture(str(video))
    pe = G.FlowPanEstimator()
    for f in tqdm(range(len(frames)), desc=f"pan {js.stem[:30]}", leave=False):
        ok, img = cap.read()
        if not ok:
            break
        if f in shots:
            pe.reset()
        r = frames[f]
        pan = pe.update(img, [r["skier"]] if r["skier"] else [])
        r["pan"] = pan
        if r["skier"]:
            r["skier_X"] = (r["skier"][0] + r["skier"][2]) / 2 + pan
    cap.release()
    # 2) skip shots that are not the boat camera (shore/replay cameras show the towing boat itself)
    if not any("boats" in r for r in frames):  # older runs did not record boats -> detect on a few frames
        cap = cv2.VideoCapture(str(video))
        for s0, s1 in zip(shots, shots[1:] + [len(frames)]):
            for f in np.linspace(s0, s1 - 1, min(5, s1 - s0)).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, img = cap.read()
                if ok:
                    frames[f]["boats"] = BOAT.predict(img, classes=[8], conf=0.35, verbose=False)[0].boxes.xyxy.tolist()
        cap.release()
        for r in frames:
            r.setdefault("boats", None)
        sub = {s0: [f for f in range(s0, s1) if frames[f]["boats"] is not None]
               for s0, s1 in zip(shots, shots[1:] + [len(frames)])}
        boat_view = {}
        for s0, fl in sub.items():
            votes = [any((b[2] - b[0]) * (b[3] - b[1]) > 0.01 * meta["width"] * meta["height"]
                         and b[3] < 0.97 * meta["height"] for b in frames[f]["boats"]) for f in fl]
            boat_view[s0] = not (votes and np.mean(votes) >= 0.5)
    else:
        boat_view = None
    # 3) tracking + v2 course logic (shared with the pipeline)
    crs = C.analyse_course(frames, meta, boat_view)
    events, tracks_all, passes, courses = crs["events"], crs["tracks"], crs["passes"], crs["courses"]
    cs, main = crs["main_score"], crs["main"]
    m = re.search(r"(\d(?:\.\d+)?)\s*(?:at|@)", js.stem.replace("_", " "))
    official = m.group(1) if m else "?"
    res_v2 = dict(video=meta["video"], official=official, main_pass=main, scores=crs["scores"], courses=courses,
                  events=events)
    (out / f"{js.stem}.v2.json").write_text(json.dumps(res_v2, indent=1, default=float))
    rows.append(dict(video=js.stem, official=official, v1_score=res["score"].get("score", "-"),
                     estimate=cs["estimate"], confirmed=cs["confirmed"], seen=f"{cs['seen']}/6",
                     slots=C.slots_str(courses[main]) if main is not None else "",
                     gates=sum(e["kind"] == "gate" for e in events) // 2, passes=len(passes),
                     skipped_shots=sum(not v for v in crs["boat_view"].values())))
    if a.render:
        events = crs["shown_events"]
        from cvdemo.pipeline import render
        segs = [dict(start=s0, end=s1, centre=None, amp=None) for s0, s1 in zip(shots, shots[1:] + [len(frames)])]
        render(video, out / f"{js.stem}.mp4", frames, meta, events, segs, tracks_all, res["models"],
               "re-scored v2: optical-flow pan + local swing centre + duplicate merge")

hdr = ["video", "official", "v1_score", "estimate", "confirmed", "seen", "slots", "gates", "passes", "skipped_shots"]
md = ["Slots: L/R = side of the course in the image; + rounded outside, x passed inside, ? not seen, "
      "- not reached (run ended), ~ too close to call / position projected, ! only contradicting detections. Gates (passed between) are "
      "excluded from the 6 turn slots; shots that are not the boat camera (e.g. shore replays) are skipped.", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + \
     ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
