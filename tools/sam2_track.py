"""SAM 2 video segmentation/tracking of the skier and the buoys on a short clip.

The first frame is prompted automatically with boxes (skier from YOLO11, buoys from the
chosen buoy detector); SAM 2 then propagates the masks through the clip (memory attention),
which keeps buoys locked even when the detector loses them in spray.

    python tools/sam2_track.py VIDEO --start 9.0 --dur 4 --model sam2.1_t.pt --buoy gdino --out runs/sam2
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

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0)
ap.add_argument("--dur", type=float, default=4)
ap.add_argument("--model", default="sam2.1_t.pt")
ap.add_argument("--buoy", default="gdino")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--out", default=str(ROOT / "runs/sam2"))
ap.add_argument("--tag")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = a.tag or f"{Path(a.video).stem.replace(' ', '_')}_{Path(a.model).stem}"
clip = out / f"{tag}_clip.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start), "-i", a.video, "-t", str(a.dur),
                "-an", "-c:v", "libx264", "-crf", "16", str(clip)], check=True)

cap = cv2.VideoCapture(str(clip))
ok, f0 = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.sam import SAM2VideoPredictor  # noqa: E402

t = time.perf_counter()
pr = YOLO(str(ROOT / "models/yolo11s.pt")).predict(f0, classes=[0], conf=0.25, verbose=False)[0]
persons = pr.boxes.xyxy.tolist()
skier_box = persons[int(pr.boxes.conf.argmax())] if persons else None
buoy_boxes = [d[:4] for d in B.build(a.buoy)(f0, persons)]
t_prompt = time.perf_counter() - t
prompts = ([skier_box] if skier_box else []) + buoy_boxes
names = (["skier"] if skier_box else []) + [f"buoy{i + 1}" for i in range(len(buoy_boxes))]
print("prompts:", list(zip(names, [[round(v) for v in b] for b in prompts])))
if not prompts:
    sys.exit("nothing to track in the first frame")

pred = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=a.imgsz,
                                         model=str(ROOT / "models" / a.model), save=False, verbose=False))
cols = [(255, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 128, 0)]
H, W = f0.shape[:2]
tmp = str(out / f"{tag}.tmp.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
per_frame, areas = [], []
t0 = time.perf_counter()
last = t0
for r in pred(source=str(clip), bboxes=prompts, labels=[1] * len(prompts), stream=True):
    now = time.perf_counter()
    per_frame.append(now - last)
    last = now
    img = r.orig_img.copy()
    present = []
    if r.masks is not None:
        m = r.masks.data.cpu().numpy()
        for i in range(min(len(m), len(names))):
            mk = cv2.resize(m[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            present.append(int(mk.sum()))
            if mk.sum() == 0:
                continue
            c = np.array(cols[i % len(cols)], np.uint8)
            img[mk] = (0.45 * img[mk] + 0.55 * c).astype(np.uint8)
            ys, xs = np.where(mk)
            cv2.putText(img, names[i], (int(xs.min()), int(ys.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        tuple(int(v) for v in c), 1, cv2.LINE_AA)
    areas.append(present)
    cv2.putText(img, f"SAM 2 ({Path(a.model).stem}) video propagation, {per_frame[-1] * 1000:.0f} ms/frame CPU",
                (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    vw.write(img)
vw.release()
total = time.perf_counter() - t0
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out / f"{tag}.mp4")], check=True)
Path(tmp).unlink()
clip.unlink()
steady = per_frame[1:] or per_frame
res = dict(video=a.video, start=a.start, dur=a.dur, model=a.model, imgsz=a.imgsz, objects=names,
           frames=len(per_frame), total_s=round(total, 2), prompt_detection_s=round(t_prompt, 2),
           ms_per_frame=round(1000 * float(np.median(steady)), 1),
           x_realtime=round(total / (len(per_frame) / fps), 2),
           mask_kept_ratio=[round(float(np.mean([(p[i] if i < len(p) else 0) > 0 for p in areas])), 3)
                            for i in range(len(names))])
(out / f"{tag}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
