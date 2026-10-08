"""Joint ("rig") calibration: one lens + camera height shared by several videos from the same boat camera.

Single-video calibration is under-determined (lens, height, tilt and scale trade off). Videos filmed with the
same boat camera share the lens (focal length relative to image width) and the camera height, so we grid-search
those two numbers and fit only tilt, pan offset/drift and boat speed per video. The pair that explains all
videos best (lowest total robust cost) is the rig calibration.

    python tools/calibrate_rig.py --frames-dir <dir with *.frames.json> --only "Damir" "Lucas" "Vincenzo" \
        --out results/metric_v3/rig_italy.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import camera as CAM  # noqa: E402
from cvdemo import course3 as C3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "results/colab_t4/demo"))
ap.add_argument("--frames-dir")
ap.add_argument("--only", nargs="+", required=True, help="substrings of the videos filmed with this camera")
ap.add_argument("--f-grid", nargs="+", type=float, default=[0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75])
ap.add_argument("--h-grid", nargs="+", type=float, default=[1.6, 1.9, 2.2, 2.5, 2.8, 3.1])
ap.add_argument("--cache", default=str(ROOT / "runs/rig_cache.pkl"))
ap.add_argument("--out", default=str(ROOT / "results/metric_v3/rig.json"))
a = ap.parse_args()
fdir = Path(a.frames_dir or a.run)

# ---- per-video inputs (cached: the optical-flow pass is the slow part)
cache = Path(a.cache)
data = pickle.loads(cache.read_bytes()) if cache.exists() else {}
from ultralytics import YOLO  # noqa: E402

boat = None
for js in sorted(Path(a.run).glob("*.json")):
    if js.name.endswith(".frames.json") or js.stem == "summary" or not any(o in js.stem for o in a.only):
        continue
    if js.stem in data:
        continue
    boat = boat or YOLO(str(ROOT / "models/yolo11s.pt"))
    meta = json.loads(js.read_text())["meta"]
    frames = json.loads((fdir / f"{js.stem}.frames.json").read_text())
    video = ROOT / meta["video"].split("CV_test/")[-1]
    inc, boat_view = C3.prepare_video(video, frames, meta, boat)
    shots = list(zip(meta["shots"], meta["shots"][1:] + [len(frames)]))
    s0, s1 = max((s for s in shots if boat_view[s[0]]), key=lambda s: s[1] - s[0])
    skier_obs, ptracks = C3.calibration_inputs(frames, meta, s0, s1)
    data[js.stem] = dict(meta=meta, inc=inc, s=(s0, s1), skier=skier_obs, tracks=ptracks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(data))
names = [n for n in data if any(o in n for o in a.only)]
print("videos:", names)


def fit_one(name, fixed, quick=True):
    d = data[name]
    m = d["meta"]
    cal = CAM.ShotCalibration(d["s"][0], d["s"][1], m["width"], m["height"], m["fps"], d["inc"])
    cal.fit(d["skier"], d["tracks"], CAM.rope_length_from_name(name), v_prior=15.3 if C3.is_women(name) else 16.1,
            fixed=fixed, quick=quick)
    return cal


# ---- grid search over the shared lens + height
grid = [(fr, h) for fr in a.f_grid for h in a.h_grid]
costs = {}
for fr, h in tqdm(grid, desc="rig grid (lens x height)"):
    tot = 0.0
    for n in names:
        cal = fit_one(n, (fr, h))
        # normalise by residual count so long and short videos weigh alike
        tot += (cal.report.get("cost", 1e6) / max(1, cal.report.get("n_residuals", 1))) if cal.p is not None else 1e3
    costs[(fr, h)] = tot / len(names)
best = min(costs, key=costs.get)
print("best rig: f/W=%.2f (hfov %.0f deg), h=%.1f m, mean cost %.3f" % (
    best[0], np.degrees(2 * np.arctan(0.5 / best[0])), best[1], costs[best]))
reports = {}
for n in names:
    cal = fit_one(n, best, quick=False)
    reports[n] = cal.report
    print(f"  {n}: {json.dumps(cal.report)}")
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(f_rel=best[0], cam_height_m=best[1], videos=names,
                               grid={f"{k[0]}x{k[1]}": round(v, 4) for k, v in costs.items()},
                               per_video=reports), indent=1))
print("saved", out)
