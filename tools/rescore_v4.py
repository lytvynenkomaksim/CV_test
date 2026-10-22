"""Compare course logic v4 (skier movement + buoy proximity) with v2 (local swing centre) on saved detections.

    python tools/rescore_v4.py --run results/colab_t4/demo --frames-dir <dir with *.frames.json> \
        --out results/v4 [--only Alexandra] [--render]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import course as C2  # noqa: E402
from cvdemo import course4 as C4  # noqa: E402
from cvdemo import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--frames-dir")
ap.add_argument("--out", default=str(ROOT / "results/v4"))
ap.add_argument("--only", nargs="*")
ap.add_argument("--render", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fdir = Path(a.frames_dir or a.run)

from ultralytics import YOLO  # noqa: E402

BOAT = YOLO(str(ROOT / "models/yolo11s.pt"))
rows = []
for js in tqdm(sorted(p for p in Path(a.run).glob("*.json") if not p.name.endswith(".frames.json")
                      and p.stem != "summary"), desc="videos"):
    if a.only and not any(o in js.stem for o in a.only):
        continue
    res = json.loads(js.read_text())
    meta = res["meta"]
    frames = json.loads((fdir / f"{js.stem}.frames.json").read_text())
    video = ROOT / meta["video"].split("CV_test/")[-1]
    shots, n, fps = meta["shots"], len(frames), meta["fps"]
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    pe = G.FlowPanEstimator()
    for f in range(n):
        ok, img = cap.read()
        if not ok:
            break
        if f in shots:
            pe.reset()
        r = frames[f]
        r["pan"] = pe.update(img, [r["skier"]] if r["skier"] else [])
        if r["skier"]:
            r["skier_X"] = (r["skier"][0] + r["skier"][2]) / 2 + r["pan"]
    boat_view = {}
    for s0, s1 in zip(shots, shots[1:] + [n]):
        votes = []
        for f in np.linspace(s0, s1 - 1, min(5, s1 - s0)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if ok:
                bx = BOAT.predict(img, classes=[8], conf=0.35, verbose=False)[0].boxes.xyxy.tolist()
                votes.append(any((b[2] - b[0]) * (b[3] - b[1]) > 0.01 * meta["width"] * meta["height"]
                                 and b[3] < 0.97 * meta["height"] for b in bx))
        boat_view[s0] = not (votes and np.mean(votes) >= 0.5)
    cap.release()
    t_motion = time.perf_counter() - t0
    t0 = time.perf_counter()
    v4 = C4.analyse_course_v4(frames, meta, boat_view)
    t_v4 = time.perf_counter() - t0
    v2 = C2.analyse_course(frames, meta, boat_view)
    m = re.search(r"(\d(?:\.\d+)?)\s*(?:at|@)", js.stem.replace("_", " "))

    def slots(r):
        return C2.slots_str(r["courses"][r["main"]]) if r["main"] is not None else ""
    gates = [e for e in v4["events"] if e["kind"] == "gate"]
    rows.append(dict(video=js.stem, official=m.group(1) if m else "?",
                     v2_confirmed=v2["main_score"]["confirmed"], v2_seen=f"{v2['main_score']['seen']}/6",
                     v2_slots=slots(v2),
                     v4_confirmed=v4["main_score"]["confirmed"], v4_seen=f"{v4['main_score']['seen']}/6",
                     v4_slots=slots(v4),
                     v4_gates=" ".join(g["role"] + (":in" if g["through_gate"] else ":OUT") for g in gates
                                  if g["role"] != "guide") or "-",
                     v4_guide=sum(e["kind"] == "centre" for e in v4["events"]) + sum(
                         e["kind"] == "gate" and e.get("role") == "guide" for e in v4["events"]),
                     motion_s=round(t_motion, 1), logic_s=round(t_v4, 2)))
    (out / f"{js.stem}.v4.json").write_text(json.dumps(dict(events=v4["events"], courses=v4["courses"]),
                                                       indent=1, default=float))
    if a.render:
        from cvdemo.pipeline import render
        segs = [dict(start=s0, end=s1, centre=None, amp=None) for s0, s1 in zip(shots, shots[1:] + [n])]
        render(video, out / f"{js.stem}_v4.mp4", frames, meta, v4["shown_events"], segs, v4["tracks"],
               res["models"], "course logic v4: skier movement + buoy proximity (no centre line, no 3D)")
hdr = list(rows[0]) if rows else []
md = ["v2 = side from the skier's local swing centre; v4 = side from the skier's movement direction, pass judged "
      "on the frames where the ski is level with the buoy (in skier heights). + outside (correct), x inside, "
      "~ too close / hidden, ? not seen, - not reached. Gates: in = skier passed between the pair.", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
