"""SAM 2 segmentation of the skier and buoys, guided by the detector + tracker.

Two modes:
  guided (default) - every frame, the skier box (YOLO11) and every tracked buoy box (fast YOLO buoy
                     detector + pan-compensated tracker, including *projected* boxes of hidden buoys)
                     are passed to SAM 2 as box prompts; SAM 2 only refines them into masks.
                     A buoy mask is rejected (-> "hidden, estimated position") if it is empty, much
                     bigger than the box, or its centre drifts away from the expected position.
                     This fixes the drift of plain SAM 2 tracking (mask jumping to the ski tip).
  memory           - original SAM 2 video mode: prompt boxes only on the first frame, then SAM 2's
                     memory propagates them (drifts on tiny buoys; kept for comparison).

    python tools/sam2_track.py VIDEO --start 9 --dur 4 [--mode guided|memory] [--model sam2.1_t.pt]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo import buoys as B  # noqa: E402
from cvdemo import geometry as G  # noqa: E402
from cvdemo.pipeline import SkierTracker, _dashed_circle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--dur", type=float, default=4)
ap.add_argument("--mode", default="guided", choices=["guided", "memory"])
ap.add_argument("--model", default="sam2.1_t.pt")
ap.add_argument("--buoy", default="yolo-buoy")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=str(ROOT / "runs/sam2"))
ap.add_argument("--tag")
a = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(a.threads)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"{Path(a.video).stem.replace(' ', '_')}_{Path(a.model).stem}_{a.mode}"
clip = out / f"{tag}_clip.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start), "-i", a.video, "-t", str(a.dur),
                "-an", "-c:v", "libx264", "-crf", "16", str(clip)], check=True)
cap = cv2.VideoCapture(str(clip))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
H, W = frames[0].shape[:2]
SKIER, BUOY, HID = (255, 0, 255), (0, 165, 255), (0, 220, 255)


def overlay(img, mk, col, label=None):
    c = np.array(col, np.uint8)
    img[mk] = (0.45 * img[mk] + 0.55 * c).astype(np.uint8)
    if label:
        ys, xs = np.where(mk)
        cv2.putText(img, label, (int(xs.min()), int(ys.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    cv2.LINE_AA)


tmp = str(out / f"{tag}.tmp.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
stats = dict(frames=len(frames), buoy_boxes=0, buoy_masks_accepted=0, buoy_hidden=0, skier_masks=0)
t_det, t_sam = [], []
t0 = time.perf_counter()

if a.mode == "guided":
    from ultralytics import SAM

    sam = SAM(str(ROOT / "models" / a.model))
    skier = SkierTracker("s")
    det = B.build(a.buoy)
    pan_est = G.PanEstimator()
    tracker = B.BuoyTracker(max_miss=int(2 * fps), frame_size=(W, H))
    hid_run = {}
    for f, img0 in enumerate(frames):
        t = time.perf_counter()
        pan = pan_est.update(img0)
        best, persons = skier(img0)
        tracker.update(f, det(img0, persons), pan, img0)
        t_det.append(time.perf_counter() - t)
        prompts, kinds = [], []
        if best:
            prompts.append([float(v) for v in best[0]])
            kinds.append(("skier", None, None))
        for tr in tracker.tracks:
            if tr.n_obs() < 2 or tr.last[0] != f:
                continue
            _, X, y, w, h, s, observed = tr.last
            x = X - pan
            pad = 3 + (0 if observed else 0.6 * tr.misses)  # search a bit wider while hidden
            prompts.append([x - w / 2 - pad, y - h - pad, x + w / 2 + pad, y + pad])
            kinds.append(("buoy", tr, (x, y - h / 2, w, h, observed)))
        img = img0.copy()
        t = time.perf_counter()
        masks = []
        if prompts:
            r = sam(img0, bboxes=prompts, imgsz=a.imgsz, verbose=False)[0]
            if r.masks is not None:
                masks = [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                         for m in r.masks.data.cpu().numpy()]
        t_sam.append(time.perf_counter() - t)
        for i, (kind, tr, geo) in enumerate(kinds):
            mk = masks[i] if i < len(masks) else None
            if kind == "skier":
                if mk is not None and mk.any():
                    overlay(img, mk, SKIER, "skier")
                    stats["skier_masks"] += 1
                continue
            stats["buoy_boxes"] += 1
            cx, cy, w, h, observed = geo
            ok = False
            if mk is not None and mk.any():
                ys, xs = np.where(mk)
                area, box_area = mk.sum(), max(1.0, (w + 6) * (h + 6))
                ok = area <= 3.0 * box_area and np.hypot(xs.mean() - cx, ys.mean() - cy) <= max(w, h) + 6
            if ok and observed:
                overlay(img, mk, BUOY, f"buoy {tr.tid}")
                stats["buoy_masks_accepted"] += 1
                hid_run[tr.tid] = 0
            else:  # covered by spray / wave or mask rejected -> projected position
                stats["buoy_hidden"] += 1
                hid_run[tr.tid] = hid_run.get(tr.tid, 0) + 1
                rad = int(max(w, h) / 2 + 4 + 0.6 * hid_run[tr.tid])
                _dashed_circle(img, (int(cx), int(cy)), rad, HID)
                cv2.drawMarker(img, (int(cx), int(cy)), HID, cv2.MARKER_CROSS, 8, 1)
                cv2.putText(img, f"buoy {tr.tid} hidden (est.)", (int(cx) - rad, int(cy) - rad - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, HID, 1, cv2.LINE_AA)
        cv2.putText(img, f"detector-guided SAM 2 ({Path(a.model).stem}): det+track {t_det[-1] * 1000:.0f} ms, "
                         f"SAM {t_sam[-1] * 1000:.0f} ms / frame (CPU)", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
else:
    from ultralytics import YOLO
    from ultralytics.models.sam import SAM2VideoPredictor

    pr = YOLO(str(ROOT / "models/yolo11s.pt")).predict(frames[0], classes=[0], conf=0.25, verbose=False)[0]
    persons = pr.boxes.xyxy.tolist()
    prompts = ([persons[int(pr.boxes.conf.argmax())]] if persons else []) + \
              [d[:4] for d in B.build(a.buoy)(frames[0], persons)]
    names = (["skier"] if persons else []) + [f"buoy{i + 1}" for i in range(len(prompts) - bool(persons))]
    pred = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=a.imgsz,
                                             model=str(ROOT / "models" / a.model), save=False, verbose=False))
    last = time.perf_counter()
    for r in pred(source=str(clip), bboxes=prompts, labels=[1] * len(prompts), stream=True):
        t_sam.append(time.perf_counter() - last)
        img = r.orig_img.copy()
        if r.masks is not None:
            for i, m in enumerate(r.masks.data.cpu().numpy()[:len(names)]):
                mk = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                if mk.any():
                    overlay(img, mk, SKIER if names[i] == "skier" else BUOY, names[i])
        cv2.putText(img, f"SAM 2 memory propagation ({Path(a.model).stem}), {t_sam[-1] * 1000:.0f} ms/frame CPU",
                    (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        last = time.perf_counter()

vw.release()
total = time.perf_counter() - t0
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out / f"{tag}.mp4")], check=True)
Path(tmp).unlink()
clip.unlink()
med = lambda v: round(1000 * float(np.median(v[1:] or v)), 1) if v else None
res = dict(video=a.video, start=a.start, dur=a.dur, mode=a.mode, model=a.model, imgsz=a.imgsz, **stats,
           ms_per_frame_detect_track=med(t_det), ms_per_frame_sam=med(t_sam), total_s=round(total, 1),
           x_realtime=round(total / (len(frames) / fps), 1))
(out / f"{tag}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
