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
