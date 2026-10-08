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
ap.add_argument("--rig", help="rig calibration json from tools/calibrate_rig.py (shared lens + camera height)")
ap.add_argument("--both", action="store_true", help="render with AND without course lines (<name>_lines.mp4 / _clean.mp4)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fdir = Path(a.frames_dir or a.run)

from ultralytics import YOLO  # noqa: E402

BOAT = YOLO(str(ROOT / "models/yolo11s.pt"))
rig, rig_videos = None, set()
if a.rig:
    rj = json.loads(Path(a.rig).read_text())
    rig, rig_videos = (rj["f_rel"], rj["cam_height_m"]), set(rj["videos"])
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
    import time
    t0 = time.perf_counter()
    inc, boat_view = C3.prepare_video(video, frames, meta, BOAT)
    t_motion = time.perf_counter() - t0
    t0 = time.perf_counter()
    ana = C3.analyse_course_metric(frames, meta, inc, js.stem, boat_view, rig=rig if (
        rig and js.stem in rig_videos) else None)
    t_course = time.perf_counter() - t0
    metric_ok = any(sh.get("course") and sh["course"].get("slots") for sh in ana["shots"])
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
    det_s = res.get("processing_s")
    n, fps = meta["frames"], meta["fps"]
    times = dict(detect_s=det_s, camera_motion_s=round(t_motion, 1), course_logic_s=round(t_course, 1))
    timing_line = (f"processing: detection {det_s / n * 1000:.0f} ms/frame + camera/course "
                   f"{(t_motion + t_course) / n * 1000:.0f} ms/frame" if det_s else None)
    if a.render or a.both:
        t0 = time.perf_counter()
        if metric_ok:
            variants = [("_lines", True), ("_clean", False)] if a.both else [("", True)]
            for suf, show in variants:
                C3.render_metric(video, out / f"{js.stem}{suf}.mp4", frames, meta, ana, res["models"],
                                 "course logic v3 (metric)", show_lines=show, timing_line=timing_line)
        else:  # camera could not be calibrated: image-space (2D) course logic, no course lines possible
            from cvdemo import course as C2
            from cvdemo.pipeline import render as render2d
            crs = C2.analyse_course(frames, meta, boat_view)
            segs = [dict(start=s0, end=s1, centre=None, amp=None) for s0, s1 in zip(shots, shots[1:] + [n])]
            render2d(video, out / f"{js.stem}_2d.mp4", frames, meta, crs["shown_events"], segs, crs["tracks"],
                     res["models"], (timing_line or "") + " | camera not calibrated: 2D logic, no course lines")
            m2 = crs["main_score"]
            rows[-1].update(estimate=m2["estimate"], confirmed=m2["confirmed"], seen=f"{m2['seen']}/6",
                            slots=C2.slots_str(crs["courses"][crs["main"]]) if crs["main"] is not None else "",
                            camera="not calibrated -> 2D fallback")
        times["render_s"] = round((time.perf_counter() - t0) / (2 if (a.both and metric_ok) else 1), 1)
    rows[-1].update(frames=n, video_s=round(n / fps, 1), **times)
    if det_s:
        tot = det_s + t_motion + t_course
        rows[-1].update(total_s=round(tot, 1), s_per_video_s=round(tot / (n / fps), 1))

hdr = list(rows[0]) if rows else []
md = ["Course logic v3 (metric water-plane model). Slots: + rounded outside, x inside, ~ too close / hidden, "
      "? not seen. camera = calibration status of the main shot; turn_offset_m = measured distance of turn buoys "
      "from the centre line (true value 11.5 m).", "",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows]
(out / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
