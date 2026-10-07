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
a = ap.parse_args()

m = YOLO(str(ROOT / "models" / a.model))
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device="cpu",
        project=str(ROOT / "runs/train"), name="buoy", exist_ok=True, patience=15,
        # small objects: keep scale aug mild, no vertical flips (water plane), keep mosaic
        scale=0.3, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=5, hsv_h=0.02, plots=True)
best = ROOT / "runs/train/buoy/weights/best.pt"
dst = ROOT / "models" / f"buoy_{Path(a.model).stem}.pt"
shutil.copy(best, dst)
print("saved", dst)
YOLO(str(dst)).export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True)
