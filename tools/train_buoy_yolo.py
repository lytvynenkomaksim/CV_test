"""Train a fast YOLO11 buoy detector on Grounding-DINO pseudo labels ("distillation").

    python tools/train_buoy_yolo.py --data data/buoy_dataset/data.yaml --model yolo11s.pt --epochs 40

Copies the best weights to models/buoy_yolo11s.pt and exports ONNX (for TensorRT / DeepStream).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data/buoy_dataset/data.yaml"))
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=1088)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--device", default=None, help="cpu, 0 (first GPU), ... default: GPU if available")
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--cache", default=False, help="ram | disk | False")
ap.add_argument("--name", default=None, help="run name under runs/train (default buoy_<model>)")
a = ap.parse_args()

# data.yaml keeps a relative path; resolve it so the dataset works wherever the repo is cloned
data = Path(a.data).resolve()
cfg = data.read_text().splitlines()
cfg = [f"path: {data.parent}" if l.startswith("path:") else l for l in cfg]
run_yaml = data.parent / "data_resolved.yaml"
run_yaml.write_text("\n".join(cfg) + "\n")
if a.device is None:
    import torch
    a.device = 0 if torch.cuda.is_available() else "cpu"

run_name = a.name or f"buoy_{Path(a.model).stem}"
m = YOLO(str(ROOT / "models" / a.model))
m.train(data=str(run_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
        project=str(ROOT / "runs/train"), name=run_name, exist_ok=True, patience=a.patience,
        cache=(a.cache if a.cache in ("ram", "disk") else False),
        # small objects: keep scale aug mild, no vertical flips (water plane), keep mosaic
        scale=0.3, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=5, hsv_h=0.02, plots=True)
best = ROOT / "runs/train" / run_name / "weights/best.pt"
dst = ROOT / "models" / f"buoy_{Path(a.model).stem}.pt"
shutil.copy(best, dst)
print("saved", dst)
YOLO(str(dst)).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)
