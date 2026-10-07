# NVIDIA DeepStream deployment (prepared, not yet run)

**Status:** these configs and `ds_pipeline.py` are written but **untested**. The development sandbox has
no NVIDIA GPU, and DeepStream can't run on CPU.

## What DeepStream adds, and what it doesn't

DeepStream is NVIDIA's video-analytics *framework* (GStreamer + TensorRT), not a model. It runs the
**same detectors** we already trained (YOLO11/YOLO26/RF-DETR exported to ONNX → TensorRT). So it
changes **speed and scale, not accuracy**:

| Stage | Python pipeline (`cvdemo`) | DeepStream |
|---|---|---|
| Video decode | CPU (OpenCV/FFmpeg) | GPU hardware decoder (NVDEC) |
| Detection | PyTorch | TensorRT FP16/INT8 engine |
| Tracking | ByteTrack / BoT-SORT (Python) | NvDCF / NvDeepSORT (C++/CUDA) |
| Many cameras | one video at a time | dozens of streams batched on one GPU |
| Drawing + encode | CPU | GPU (nvdsosd + NVENC) |
| Course / buoy-pass logic | `cvdemo/course.py` | same code, run on DeepStream's output (frames.json) |

Rough expectation, not yet measured: per-frame detection drops from ~25 ms (PyTorch, T4) to roughly
3–8 ms (TensorRT FP16). This is mainly worth it for **live, multi-camera** use. For offline analysis
of single videos, exporting the YOLO model to TensorRT (`format=engine`, done in
`colab/train_v2.ipynb`) gives most of the speed-up with far less setup.

## Requirements

- An NVIDIA GPU (T4/L4/A10/A100, or a Jetson Orin), Linux, and Docker with the NVIDIA container toolkit.
- The DeepStream 7.x container, e.g. `nvcr.io/nvidia/deepstream:7.1-gc-triton-devel` (free NGC account).
- **Google Colab can't run it:** no Docker, and installing the SDK natively needs matching
  Ubuntu/driver versions. Use a cloud GPU VM (GCP/AWS/Lambda) or a local NVIDIA machine.

## Steps

```bash
# 1. start the container in the repo root
docker run --gpus all -it --rm -v $PWD:/work -w /work/deploy/deepstream nvcr.io/nvidia/deepstream:7.1-gc-triton-devel

# 2. build the YOLO output parser (works for YOLO11 and YOLO26)
git clone https://github.com/marcoslucianops/DeepStream-Yolo
cd DeepStream-Yolo && CUDA_VER=12.6 make -C nvdsinfer_custom_impl_Yolo && cd ..

# 3. export ONNX in the layout the parser expects (DeepStream-Yolo/utils/export_yolo11.py / export_yolo26.py)
python3 DeepStream-Yolo/utils/export_yolo11.py -w ../../weights/buoy_yolo11s.pt -s 1280 --simplify   # -> buoy_yolo11s.onnx
python3 DeepStream-Yolo/utils/export_yolo11.py -w ../../models/yolo11s.pt -s 960 --simplify          # -> yolo11s.onnx (person)

# 4. run (the TensorRT engines are built on the first run, which takes a few minutes)
pip install pyds   # or use the container's bindings
python3 ds_pipeline.py ../../data/videos/youtube/youtube_regina_jaquess.mp4 --out ../../runs/deepstream

# 5. course logic + annotated video on top of DeepStream's detections (CPU)
python3 ../../tools/rescore.py --run ../../runs/deepstream --render
```

## Notes

- **Skier detector alternative:** NVIDIA **PeopleNet** (TAO, on NGC) is an NVIDIA-licensed person
  detector that avoids the Ultralytics AGPL licence. Swap `config_infer_person.txt` for the PeopleNet
  config that ships with DeepStream.
- **Skeleton:** DeepStream has a body-pose sample (BodyPose2D/3D from TAO). Our YOLO-pose can also run
  as a secondary `nvinfer` with a custom parser. Not included here.
- **Scene cuts and camera pan** are not computed inside DeepStream. `tools/rescore.py` recomputes the pan
  from the video, but the DeepStream output is written as one shot, so cut detection is missing. Port the
  histogram cut detector from `cvdemo/geometry.py` if broadcast edits matter.
- **Licences:** DeepStream itself is free to use under NVIDIA's licence. The model licences still apply
  (Ultralytics YOLO = AGPL-3.0 or an enterprise licence; RF-DETR = Apache-2.0).
