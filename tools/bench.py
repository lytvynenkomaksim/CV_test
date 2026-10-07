"""Per-model latency benchmark on real frames (CPU), to estimate processing cost.

    python tools/bench.py VIDEO [VIDEO ...] --frames 20 --threads 4 --out data/benchmarks/bench.json

For every video, K frames are sampled evenly; each model is warmed up once and then timed
on all K frames. Pose models are timed per person crop (skier box from YOLO11s).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--frames", type=int, default=20)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--only", nargs="*", help="subset of model keys")
ap.add_argument("--out", default=str(ROOT / "data/benchmarks/bench.json"))
a = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(a.threads)
cv2.setNumThreads(a.threads)
from ultralytics import RTDETR, YOLO  # noqa: E402

from cvdemo import buoys as B  # noqa: E402
from cvdemo import pose as P  # noqa: E402

M = ROOT / "models"


def sample(video, k):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, n - 1, k + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    return out


def person_fn(path, imgsz=960, cls=YOLO):
    m = cls(str(path))
    return lambda f: m.predict(f, classes=[0], conf=0.2, imgsz=imgsz, verbose=False)[0]


DET = {
    "person:yolo11n": lambda: person_fn(M / "yolo11n.pt"),
    "person:yolo11s": lambda: person_fn(M / "yolo11s.pt"),
    "person:yolo11m": lambda: person_fn(M / "yolo11m.pt"),
    "person:rtdetr-l": lambda: person_fn(M / "rtdetr-l.pt", 640, RTDETR),
    "buoy:yolo11n-buoy": lambda: B.YoloBuoy(str(M / "buoy_yolo11n.pt")),
    "buoy:yolo11s-buoy": lambda: B.YoloBuoy(str(M / "buoy_yolo11s.pt")),
    "buoy:yolo-world-m": lambda: B.YoloWorldBuoy(),
    "buoy:grounding-dino-t": lambda: B.GDinoBuoy(),
    "buoy:owlv2-b": lambda: B.Owlv2Buoy(),
}
POSE = {f"pose:{k}": (lambda k=k: P.build(k)) for k in ["yolo-n", "yolo-s", "yolo-m", "vitpose", "rtmpose", "mediapipe"]}

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
results = dict(machine=dict(cpu=platform.processor() or platform.machine(), threads=a.threads,
                            torch=torch.__version__, gpu=gpu), videos={})
skier = YOLO(str(M / "yolo11s.pt"))
for v in a.videos:
    frames = sample(v, a.frames)
    H, W = frames[0].shape[:2]
    boxes = []
    for f in frames:
        r = skier.predict(f, classes=[0], conf=0.2, imgsz=960, verbose=False)[0]
        boxes.append(r.boxes.xyxy[r.boxes.conf.argmax()].tolist() if len(r.boxes) else None)
    vres = dict(resolution=f"{W}x{H}", frames=len(frames), models={})
    for key, mk in {**DET, **POSE}.items():
        if a.only and key not in a.only and key.split(":")[0] not in a.only:
            continue
        try:
            t = time.perf_counter()
            fn = mk()
            load = time.perf_counter() - t
        except Exception as e:  # missing weights etc.
            vres["models"][key] = dict(error=str(e)[:200])
            continue
        is_pose = key.startswith("pose")
        items = [(f, b) for f, b in zip(frames, boxes) if b is not None] if is_pose else [(f, None) for f in frames]
        call = (lambda f, b: fn(f, b)) if is_pose else (lambda f, b: fn(f))
        call(*items[0])  # warm-up
        ts, outs = [], []
        for f, b in items:
            if gpu:
                torch.cuda.synchronize()
            t = time.perf_counter()
            o = call(f, b)
            if gpu:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
            outs.append(o)
        extra = {}
        if key.startswith("buoy"):
            extra["mean_buoys_per_frame"] = round(float(np.mean([len(o) for o in outs])), 2)
            extra["frames_with_buoy"] = round(float(np.mean([len(o) > 0 for o in outs])), 2)
        if is_pose:
            extra["mean_kpt_conf"] = round(float(np.mean([o[:, 2].mean() if o is not None else 0 for o in outs])), 3)
        vres["models"][key] = dict(load_s=round(load, 2), ms_per_frame=round(1000 * float(np.median(ts)), 1),
                                   p90_ms=round(1000 * float(np.percentile(ts, 90)), 1), n=len(ts), **extra)
        print(Path(v).name[:40], key, vres["models"][key], flush=True)
        del fn
    results["videos"][v] = vres
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(results, indent=1))
