"""Run the demo pipeline on all test videos and write a summary table.

    python tools/run_all.py --buoy yolo-buoy --pose yolo-s --out runs/demo

The official result is parsed from the file name ("6 at 32 off" -> 6 buoys at 32 ft off)
and shown next to the demo's simplified score. It is NOT a validated accuracy measure:
a file may contain partial footage, replays or several passes.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="*")
ap.add_argument("--buoy", default="yolo-buoy")
ap.add_argument("--pose", default="yolo-s")
ap.add_argument("--skier", default="s")
ap.add_argument("--buoy-every", type=int, default=1)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=str(ROOT / "runs/demo"))
ap.add_argument("--skip-done", action="store_true")
ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
a = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(a.threads)
from cvdemo.pipeline import run  # noqa: E402

videos = a.videos or sorted(str(p) for p in (ROOT / "data/videos").rglob("*.mp4"))
out = Path(a.out)
rows = []
for v in videos:
    tag = Path(v).stem.replace(" ", "_")
    js = out / f"{tag}.json"
    if a.skip_done and js.exists():
        res = json.loads(js.read_text())
    else:
        res = run(v, out, a.buoy, a.pose, a.skier, a.buoy_every, tag=tag, tracker=a.tracker)
    m = re.search(r"(\d(?:\.\d+)?)\s*(?:at|@)", Path(v).stem)
    rows.append(dict(video=str(Path(v).relative_to(ROOT)) if Path(v).is_absolute() else v,
                     official=m.group(1) if m else "?", estimate=res["score"]["estimate"],
                     confirmed=res["score"]["confirmed"], seen=res["score"]["seen"], slots=res["score"]["slots"],
                     skier_visible=res["skier"]["skier_visible_ratio"], pose_good=res["skier"]["pose_good_ratio"],
                     buoy_frames=res["buoys"]["frames_with_buoy_ratio"],
                     buoy_hidden=res["buoys"]["hidden_ratio_within_tracks"],
                     hidden_gap_s=res["buoys"]["longest_hidden_gap_s"],
                     ms_per_frame=round(1000 * res["processing_s"] / res["meta"]["frames"]),
                     x_realtime=res["x_realtime"], output=Path(res["output_video"]).name))

hdr = ["video", "official", "estimate", "confirmed", "seen", "slots", "skier_visible", "pose_good",
       "buoy_frames", "buoy_hidden", "hidden_gap_s", "ms_per_frame", "x_realtime", "output"]
md = [f"# Demo run: buoy={a.buoy}, pose={a.pose}, skier=yolo11{a.skier}, buoy detector every {a.buoy_every} frame(s)",
      "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    md.append("| " + " | ".join(str(r[h]) for h in hdr) + " |")
(out / "summary.md").write_text("\n".join(md) + "\n")
(out / "summary.json").write_text(json.dumps(rows, indent=1))
print("\n".join(md))
