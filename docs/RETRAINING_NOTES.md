# Notes for the next buoy-detector training run

Agreed with the team (2026-10-07):

- **Hardware:** Colab Pro is available, so use an A100 (or L4) runtime, not a T4.
  - A100 has more CPU cores, which removes the augmentation bottleneck seen on the T4 run (~73 s/epoch).
  - Use a larger batch (e.g. 32–64 at 1280 px) and `workers` = number of CPU cores.
- **Validation:** keep validating every epoch. That's fine.
- **Speed:** `cache="ram"`, 30–40 epochs with `patience=10` (the T4 run plateaued around epoch 30–40).
- **Progress:** the notebook must show tqdm progress bars, with no `| tail` that hides output.
- **Data:** train the second model on the *cleaned* Colab dataset (1,608 frames). Do NOT drop buoys that overlap
  the skier: the skier can pass over or next to a buoy, and that is exactly the moment we need. Only reject
  detections on the skier's body (torso/arms from the skeleton). Use temporal consistency to remove
  flickering false positives (spray, steering wheel, ski tip).
- **First T4 run (baseline, raw labels):** YOLO11s @1280, best mAP50 ≈ 0.76, mAP50-95 ≈ 0.585 (epoch ~40),
  precision ≈ 0.75–0.80, recall ≈ 0.65–0.72 on 3 held-out videos.

## Prepared for the next run (2026-10-07, evening)

- `colab/train_v2.ipynb`: A100, tqdm everywhere. It trains **YOLO11s, YOLO26s and RF-DETR Small** on the
  cleaned dataset, evaluates all of them (plus the T4 baseline and Grounding DINO) on the same held-out
  videos, exports TensorRT FP16, and runs the full demo with the best model. Each step can be switched off.
- `data/buoy_dataset_v2`: the Colab dataset (1,608 frames) after `tools/clean_dataset.py`.
  - **Removed:** 221 wrong shape/size, 51 on the skier's body, 134 skier legs/ski (non-buoy colours near
    the skier), 208 flickering (no temporal support), 4 on our own boat.
  - **Recovered:** 5 real buoys next to the skier.
  - Review galleries are in `data/buoy_dataset_v2/review/`.
  - Labels only in git; `tools/extract_label_frames.py` re-cuts the images from the videos (verified
    pixel-identical).
- **Baseline on the cleaned validation set** (T4 model, raw labels): AP50 0.798, P 0.715, R 0.819,
  R_centre 0.862.
- Verified here on CPU: RF-DETR training + loading + evaluation (1-epoch smoke test), YOLO26 training +
  ONNX export.
