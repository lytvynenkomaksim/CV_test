"""Fine-tune RF-DETR (transformer detector, DINOv2 backbone, Apache-2.0) on the buoy dataset.

    pip install "rfdetr[train,loggers]"
    python tools/train_rfdetr.py --dataset data/buoy_dataset_v2 --variant RFDETRSmall --resolution 896 --epochs 40

RF-DETR reads the YOLO-format dataset (data.yaml + images/ + labels/) directly. The best checkpoint is
copied to models/buoy_rfdetr.pth; tools/eval_detectors.py and cvdemo (--buoy rfdetr) use it.
Resolution must be a multiple of the model's block size (56 is safe: 560, 672, 784, 896, 1008, 1120).
Small buoys benefit from higher resolution; 896 fits comfortably on an A100.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(ROOT / "data/buoy_dataset_v2"))
ap.add_argument("--variant", default="RFDETRSmall", help="RFDETRNano | RFDETRSmall | RFDETRMedium | RFDETRBase")
ap.add_argument("--resolution", type=int, default=896)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", default=str(ROOT / "runs/train/buoy_rfdetr"))
a = ap.parse_args()

import rfdetr  # noqa: E402

# RF-DETR resolves 'path:' in data.yaml relative to the dataset dir; make it absolute to be safe
ds = Path(a.dataset).resolve()
y = (ds / "data.yaml").read_text().splitlines()
(ds / "data.yaml").write_text("\n".join(f"path: {ds}" if l.startswith("path:") else l for l in y) + "\n")
try:
    m = getattr(rfdetr, a.variant)()
    m.train(dataset_dir=str(ds), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.grad_accum, lr=a.lr,
            resolution=a.resolution, output_dir=a.out, num_workers=a.workers, early_stopping=True,
            early_stopping_patience=10, progress_bar="tqdm", tensorboard=False)
finally:
    (ds / "data.yaml").write_text("\n".join(y) + "\n")
best = sorted(glob.glob(f"{a.out}/checkpoint_best_total.pth")) or sorted(glob.glob(f"{a.out}/checkpoint_best*.pth"))
dst = ROOT / "models/buoy_rfdetr.pth"
shutil.copy(best[0], dst)
(ROOT / "models/buoy_rfdetr.json").write_text(json.dumps(dict(variant=a.variant, resolution=a.resolution)))
print("saved", dst, "variant", a.variant, "resolution", a.resolution)
