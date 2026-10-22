"""High-quality videos, NO trained models of ours: skier = COCO YOLO11 + ByteTrack, skeleton = COCO
YOLO11-pose, buoys = YOLOE-26L "buoy" only in the zone near the skier, ski = YOLOE-26L "water ski" at the
feet; course logic v4 (skier movement + buoy proximity). Writes <out>/<name>.mp4/.json and summary.md.

    python tools/run_hq.py data/videos/drive_extra/ItalyPro_OpenWomenRd2__Julie_Mishler_6at32_Rd2.mp4 ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import buoys as B  # noqa: E402
from cvdemo import course as C2  # noqa: E402
from cvdemo import course4 as C4  # noqa: E402
from cvdemo import pipeline as PL  # noqa: E402
from cvdemo import pose as P  # noqa: E402
from cvdemo.ski import SkiDetector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--out", default=str(ROOT / "results/hq_untrained"))
ap.add_argument("--max-frames", type=int)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

timer = PL.Timer()
with timer("load_models"):
    skier = PL.SkierTracker("s")
    pose_est = P.build("yolo-s")
    buoy_det = B.build("yoloe-zone")
    ski_det = SkiDetector()
labels = dict(skier=skier.name, pose=pose_est.name, buoy=buoy_det.name, ski=ski_det.name)
import torch  # noqa: E402

dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (4 cores)"
rows = []
for v in a.videos:
    v = Path(v)
    tag = v.stem.replace(" ", "_")
    timer = PL.Timer()
    skier.reset()
    frames, meta = PL.analyse(v, buoy_det, pose_est, skier, timer=timer, max_frames=a.max_frames, ski_det=ski_det)
    with timer("course_logic"):
        crs = C4.analyse_course_v4(frames, meta)
    tsum = timer.summary()
    proc_s = sum(x["total_s"] for k, x in tsum.items() if k != "load_models")
    n, fps = meta["frames"], meta["fps"]
    timing = f"{dev}: {proc_s / n * 1000:.0f} ms/frame, {proc_s / (n / fps):.1f} s per 1 s of video"
    segs = [dict(start=s0, end=s1, centre=None, amp=None) for s0, s1 in zip(meta["shots"], meta["shots"][1:] + [n])]
    PL.render(v, out / f"{tag}.mp4", frames, meta, crs["shown_events"], segs, crs["tracks"], labels,
              timing + " | no trained models of ours")
    slots = C2.slots_str(crs["courses"][crs["main"]]) if crs["main"] is not None else ""
    gates = [e for e in crs["events"] if e["kind"] == "gate" and e.get("role") != "guide"]
    m = re.search(r"(\d(?:\.\d+)?)\s*(?:at|@)", v.stem.replace("_", " "))
    rows.append(dict(video=tag, official=m.group(1) if m else "?", slots=slots,
                     gates=" ".join(f"{g['role']}:{'in' if g['through_gate'] else 'OUT'}" for g in gates) or "-",
                     ski_found=f"{sum(1 for r in frames if r.get('ski'))}/{sum(1 for r in frames if r['skier'])}",
                     frames=n, ms_per_frame=round(proc_s / n * 1000),
                     s_per_video_s=round(proc_s / (n / fps), 1),
                     **{f"ms_{k}": round(x["total_s"] / n * 1000) for k, x in tsum.items() if k != "load_models"}))
    (out / f"{tag}.json").write_text(json.dumps(dict(meta=meta, models=labels, timings=tsum, row=rows[-1],
                                                     events=crs["events"]), indent=1, default=float))
    (out / f"{tag}.frames.json").write_text(json.dumps(frames, default=float))
    print(rows[-1], flush=True)
    hdr = list(rows[0])
    md = ["High-quality videos, no trained models of ours (YOLOE 'buoy' in the skier zone, YOLOE 'water ski' at the "
          "feet, COCO YOLO11 skier + pose), course logic v4. Slots: + outside (correct), x inside, ~ too close / "
          "hidden, ? not seen, - not reached, ! conflicting.", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r.get(h, "")) for h in hdr) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
