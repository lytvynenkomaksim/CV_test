# Water-ski slalom computer vision: demo

Front-view slalom videos, filmed from the boat looking back at the skier. The demo:
- **tracks the skier** and draws a **skeleton** (head, torso, arms, legs);
- **detects and tracks the buoys**, including an estimated position while a buoy is hidden by spray;
- decides for each turn buoy whether the skier went round it on the **outside** (right buoy on the
  right, left buoy on the left);
- recognises the **entry and exit gates**;
- fits the **6-buoy course**.

## Layout

| Path | What |
|---|---|
| `cvdemo/` | the pipeline: `pipeline.py` (run + render), `buoys.py` (detectors + tracker), `pose.py` (5 skeleton models), `geometry.py` (camera pan, cuts, pass logic), `course.py` (course-level logic shared by all tools) |
| `tools/` | `video_audit.py`, `autolabel_gdino.py`, `clean_dataset.py`, `extract_label_frames.py`, `train_buoy_yolo.py`, `train_rfdetr.py`, `eval_detectors.py`, `bench.py`, `sam2_track.py`, `run_all.py`, `rescore.py` |
| `colab/` | `cv_demo_gpu.ipynb` (first T4 run, already executed), `train_v2.ipynb` (retraining: YOLO11 vs YOLO26 vs RF-DETR + TensorRT, prepared, not run) |
| `deploy/deepstream/` | NVIDIA DeepStream configs + pyds pipeline (prepared, untested: needs an NVIDIA GPU) |
| `data/videos/` | test videos: 10 front-view + 4 extra (Drive), 1 YouTube 1080p; see `SOURCES.md` |
| `data/audit/` | Drive folder survey (which folders are front view) and per-video quality audit |
| `data/buoy_dataset/`, `data/buoy_dataset_v2/` | auto-labelled buoy datasets (v2 = cleaned; labels only, images re-cut from the videos) |
| `weights/` | trained buoy detector (`buoy_yolo11s.pt`, `.onnx`) |
| `results/` | Colab T4 run (benchmarks, per-video results, training curves), `rescore_v2/` (current course logic), `eval_baseline_cpu/` |
| `docs/` | `MODEL_LANDSCAPE.md` (models, transformers, NVIDIA, licences), `RETRAINING_NOTES.md` |

## Run

```bash
pip install -r requirements.txt && bash tools/download_models.sh && cp weights/buoy_yolo11s.pt models/
python -m cvdemo.pipeline "data/videos/front_view/Lucas Cornale 6 at 35 off.mp4" --out runs/demo   # one video
python tools/run_all.py --out runs/demo            # all videos + summary.md
python tools/rescore.py --run runs/demo --render   # re-run only the course logic on saved detections (CPU)
```

## Current results (details in `results/`)

**Speed, measured on a T4 GPU:**

| Component | Time per frame |
|---|---|
| Skier detection (YOLO11s) | 20 ms |
| Skeleton (YOLO11-pose) | 12–23 ms |
| Buoys (fine-tuned YOLO11s) | 25 ms |
| **Whole pipeline** | **~60 ms**, about 16 fps |

On the 4-core CPU in the sandbox the whole pipeline takes about 500 ms per frame.

**Detection quality:**
- **Buoy detector:** AP50 0.80 / recall 0.82 on the cleaned held-out frames.
- **Skier:** found in 77–94 % of frames.
- **Skeleton:** usable in 67–90 % of frames.

**Buoy-pass judgement** (`results/rescore_v2/summary.md`):
- **When a buoy pass is seen, the verdict is right about 87 % of the time**, checked against runs where the
  official result is 6 buoys, i.e. every buoy was rounded outside.
- **Only ~53 % of turn-buoy passes are seen**, because of spray, distance and fast camera pans.
  That is the main limitation, and it's a detection/data problem rather than a logic problem.

**Generalisation:** on the unseen YouTube lake the fine-tuned detector finds few buoys. RF-DETR (DINOv2)
and more varied training footage are the planned fixes.

## Licences

Ultralytics models are AGPL-3.0 (commercial use needs a licence). A permissive alternative stack is
described in `docs/MODEL_LANDSCAPE.md`. The test videos are third-party footage, for internal R&D only.
