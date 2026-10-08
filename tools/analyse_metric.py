"""Course logic v3 (metric water-plane model) on saved per-frame detections.

    python tools/analyse_metric.py --run results/colab_t4/demo --frames-dir <dir with *.frames.json> \
        --out results/metric_v3 [--only Lucas Damir] [--render]

Per video: camera motion (optical flow) -> per-shot camera calibration (f, height, tilt, yaw, boat speed)
-> buoys/skier in metres -> passed-buoy pruning, gates, turn buoys, 6-slot course -> summary + video.
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
from cvdemo import camera as CAM  # noqa: E402
from cvdemo import course3 as C3  # noqa: E402
from cvdemo import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--frames-dir")
ap.add_argument("--out", default=str(ROOT / "results/metric_v3"))
ap.add_argument("--only", nargs="*")
ap.add_argument("--render", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fdir = Path(a.frames_dir or a.run)

from ultralytics import YOLO  # noqa: E402

BOAT = YOLO(str(ROOT / "models/yolo11s.pt"))
SYM = dict(ok="+", miss="x", uncertain="~", unseen="?", not_reached="-", conflict="!")
rows = []
jsons = sorted(p for p in Path(a.run).glob("*.json") if not p.name.endswith(".frames.json") and p.stem != "summary")
for js in tqdm(jsons, desc="videos"):
    if a.only and not any(o in js.stem for o in a.only):
        continue
    res = json.loads(js.read_text())
    meta = res["meta"]
    frames = json.loads((fdir / f"{js.stem}.frames.json").read_text())
    video = ROOT / meta["video"].split("CV_test/")[-1]
    shots = meta["shots"]
    cap = cv2.VideoCapture(str(video))
    pe, cm, inc = G.FlowPanEstimator(), CAM.CameraMotion(), []
    for f in tqdm(range(len(frames)), desc=f"camera motion {js.stem[:28]}", leave=False):
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
    # boat-camera shots (shore / replay cameras show the towing boat)
    boat_view = {}
    for s0, s1 in zip(shots, shots[1:] + [len(frames)]):
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
    ana = C3.analyse_course_metric(frames, meta, inc, js.stem, boat_view)
    m = re.search(r"(\d(?:\.\d+)?)\s*(?:at|@)", js.stem.replace("_", " "))
    official = m.group(1) if m else "?"
    main = max((sh for sh in ana["shots"] if sh.get("course")), key=lambda sh: sum(
        sl["event"] is not None for sl in sh["course"]["slots"]), default=None)
    course = main["course"] if main else dict(slots=[])
    sc = G.course_score(course)
    cal_rep = main["calibration"] if main else (ana["shots"][0]["calibration"] if ana["shots"] else {})
    rows.append(dict(video=js.stem, official=official, estimate=sc["estimate"], confirmed=sc["confirmed"],
                     seen=f"{sc['seen']}/6", slots=" ".join(f"{s['side'][0].upper()}{SYM[s['status']]}" for s in course["slots"]),
                     gates=sum(e["kind"] == "gate" for e in ana["events"]),
                     camera=cal_rep.get("status", "-"), f_px=cal_rep.get("f_px"), cam_h_m=cal_rep.get("cam_height_m"),
                     speed_kmh=cal_rep.get("boat_speed_kmh"),
                     turn_offset_m=round(course.get("turn_offset_m", float("nan")), 1) if course.get("slots") else None))
    dump = dict(video=meta["video"], official=official, events=ana["events"],
                shots=[{k: v for k, v in sh.items() if k not in ("cal", "hist", "skier")} for sh in ana["shots"]])
    (out / f"{js.stem}.v3.json").write_text(json.dumps(dump, indent=1, default=float))
    if a.render:
        C3.render_metric(video, out / f"{js.stem}.mp4", frames, meta, ana, res["models"], "course logic v3 (metric)")

hdr = list(rows[0]) if rows else []
md = ["Course logic v3 (metric water-plane model). Slots: + rounded outside, x inside, ~ too close / hidden, "
      "? not seen. camera = calibration status of the main shot; turn_offset_m = measured distance of turn buoys "
      "from the centre line (true value 11.5 m).", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
