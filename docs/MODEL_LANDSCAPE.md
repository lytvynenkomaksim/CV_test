# Which models, for which job: mapping the 2026 landscape to this project

This note checks the team's research overview against what we found on the slalom footage.

## What we tested, and the verdict

| Job | Tested | Verdict on our footage |
|---|---|---|
| **Buoy detection** (8–15 px objects) | Grounding DINO (zero-shot), OWLv2 (zero-shot), YOLO-World (open-vocabulary), YOLO11s fine-tuned on auto-labels | The **fine-tuned YOLO11s** is about as good as Grounding DINO at about 22× the speed. YOLO-World misses tiny buoys. Grounding DINO and OWLv2 are only useful for **auto-labelling**. |
| **Skier detection** | YOLO11 n/s/m, RT-DETR-L | YOLO11s is fast (20 ms on a T4) and finds the skier in 77–94 % of frames. RT-DETR-L is 3× slower with no visible gain. |
| **Skeleton** | YOLO11-pose n/s/m, ViTPose-B, RTMPose-m, MediaPipe | **YOLO11-pose** is the best and fastest. ViTPose is 2nd. RTMPose and MediaPipe lose the skier in hard turns. |
| **Segmentation / tracking** | SAM 2.1 t/s/b (memory mode and detector-guided) | Excellent skier masks. Tiny buoys drift in memory mode; detector-guided mode fixes that. It costs 7–12× real-time on a T4 and adds nothing to the pass decision, so it's optional, for visuals and annotation only. |
| **Tracking** | ByteTrack (skier); our own pan-compensated buoy tracker | Works. The camera pan was the real problem, so we use BoT-SORT-style camera-motion compensation (optical flow + RANSAC on the background). `--tracker botsort` is available for the skier. |

## Transformer detectors: RF-DETR, D-FINE, DEIM, RT-DETR

- **RF-DETR is prepared** (`tools/train_rfdetr.py`, `colab/train_v2.ipynb`, `--buoy rfdetr`). It is the
  most promising alternative because:
  1. Its **DINOv2 backbone transfers better to new domains**. That is exactly our weak spot: the
     YOLO model barely sees buoys on the unseen YouTube lake (3 % of frames vs 43 % for Grounding DINO).
  2. It is **Apache-2.0**: no licence fee for commercial use (see the licences section).
  3. It is NMS-free, which avoids NMS mistakes when two buoys (a gate pair) are close together.

  The risk is **tiny objects**. DETRs work on 16-px patches at the input resolution, and our buoys are
  8–15 px, so we train at 896 px (vs 1280 for YOLO). The notebook measures whether that holds up; we
  don't assume it. The training code was verified here on CPU with a 1-epoch smoke test.
- **YOLO26** (current Ultralytics generation, NMS-free) is also included in the notebook. It is a
  drop-in replacement for YOLO11, and the comparison is cheap.
- **D-FINE / DEIM:** similar class to RF-DETR, academic code, more integration work. Only worth trying
  if RF-DETR looks promising but needs more accuracy.
- **RT-DETR-L:** benchmarked for the skier only; slower than YOLO11s with no gain.

## Zero-shot / foundation models (Grounding DINO, YOLO-World, OWLv2, SAM 3, Florence-2)

These are used where they're strong: **auto-labelling** (Grounding DINO → 1,608 labelled frames
→ cleaned → fine-tuned detector). **SAM 3** (concept prompt "buoy" + video tracking) could replace
Grounding DINO as the labeller and give masks too. The model is gated on Hugging Face (access
request needed), so it's a possible next step, not tested.

## NVIDIA (DeepStream / TensorRT)

- **Why it wasn't run:** DeepStream needs an NVIDIA GPU plus the DeepStream SDK container (Docker).
  The sandbox has no GPU, and Colab doesn't allow Docker.
- **It's a deployment framework, not a better model.** It runs the same ONNX/TensorRT models with GPU
  decode, batching of many streams, NvDCF tracking and GPU encode, so it improves **throughput, not
  accuracy**.
- **Prepared:**
  - `deploy/deepstream/`: nvinfer configs, the tracker setup, and a pyds pipeline that writes
    `frames.json`, so our course logic runs unchanged on its output.
  - A **TensorRT FP16** export + timing in `colab/train_v2.ipynb`. This works on Colab and gives
    NVIDIA-optimised numbers for the detector.
- **Worth it when:** live analysis, several boats/cameras, or a Jetson on the boat. For offline review
  of single videos, the TensorRT export alone is enough.

## Licences (matters if this becomes a product)

| Component | Licence | Commercial closed-source use |
|---|---|---|
| Ultralytics YOLO11 / YOLO26 / YOLO-pose / YOLO-World | AGPL-3.0 | needs an **Ultralytics Enterprise licence** (or open-source the whole service) |
| RF-DETR | Apache-2.0 | OK |
| Grounding DINO, OWLv2, SAM 2, ViTPose, RTMPose, MediaPipe | Apache-2.0 | OK (check each model's weights card) |
| NVIDIA DeepStream, TensorRT, PeopleNet | NVIDIA licences | generally free to deploy; read the terms |
| Training videos (TWBC broadcast footage, YouTube) | third-party copyright | OK for internal R&D; get permission before publishing or selling |

**A fully permissive stack is possible:**
- RF-DETR for the buoys and the skier (its COCO weights include "person");
- ViTPose or RTMPose for the skeleton, at some loss of quality on blurry footage;
- our own tracking and course logic, which is our own code.

## Recommendation

1. **Keep YOLO11s or YOLO26s for now.** Run `colab/train_v2.ipynb` when you're ready, to see whether
   RF-DETR closes the gap on new lakes.
2. **The biggest accuracy lever is still data, not architecture.** The detector sees only ~53 % of the
   turn-buoy passes (buoys are missed in spray, at distance and during fast pans) and generalises poorly to unseen lakes.
   10–20 more raw boat-camera passes from 2–3 lakes, auto-labelled + cleaned, would help every model.
3. **Use DeepStream/TensorRT only once the logic is final** and real-time or multi-camera is required.
