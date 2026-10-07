#!/usr/bin/env bash
# Download all model weights used by the demo into models/ (not committed: ~1 GB).
# Hugging Face models (Grounding DINO, OWLv2, ViTPose) are fetched automatically into
# models/hf on first use when HF_HOME=models/hf (set by the scripts' callers).
set -euo pipefail
cd "$(dirname "$0")/../models" 2>/dev/null || { mkdir -p "$(dirname "$0")/../models"; cd "$(dirname "$0")/../models"; }
U=https://github.com/ultralytics/assets/releases/download/v8.3.0
for w in yolo11n.pt yolo11s.pt yolo11m.pt yolo11n-pose.pt yolo11s-pose.pt yolo11m-pose.pt rtdetr-l.pt \
         yolov8m-worldv2.pt sam2.1_t.pt sam2.1_s.pt sam2.1_b.pt; do
  [ -f "$w" ] || curl -sSL -o "$w" "$U/$w"
done
mkdir -p clip
[ -f clip/ViT-B-32.pt ] || curl -sSL -o clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
G=https://storage.googleapis.com/mediapipe-models/pose_landmarker
[ -f pose_landmarker_heavy.task ] || curl -sSLO $G/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
mkdir -p rtm && cd rtm
Z=rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
[ -f "$Z" ] || { curl -sSLO https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/$Z && unzip -oq $Z; }
echo "done. The distilled buoy detector (buoy_yolo11s.pt) is in weights/ in the repo; copy it here:"
echo "  cp ../weights/buoy_yolo11*.pt ."
