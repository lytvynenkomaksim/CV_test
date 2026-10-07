"""Skeleton (2D pose) estimators behind one interface.

Every backend returns COCO-17 keypoints for one person box:
    np.ndarray (17, 3) -> x, y, score   (image coordinates)

Backends:
  yolo     - Ultralytics YOLO11-pose (top-down free, runs on the whole frame / crop)
  vitpose  - ViTPose (HF transformers, top-down on the person box)
  rtmpose  - RTMPose-m (OpenMMLab, ONNX via rtmlib)
  mediapipe- MediaPipe Pose Landmarker (33 pts, mapped to COCO-17)
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"

# COCO-17 indices
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(17)

# Simple stick figure: head, torso, arms, legs
LIMBS = {
    "torso": [(LSH, RSH), (LHIP, RHIP), (LSH, LHIP), (RSH, RHIP)],
    "arms": [(LSH, LEL), (LEL, LWR), (RSH, REL), (REL, RWR)],
    "legs": [(LHIP, LKN), (LKN, LAN), (RHIP, RKN), (RKN, RAN)],
}
COLORS = {"torso": (0, 255, 255), "arms": (255, 160, 0), "legs": (0, 200, 0), "head": (255, 0, 255)}


def crop_box(box, W, H, pad=0.25):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - pad * w)), min(W, int(x1 + pad * w))
    y0, y1 = max(0, int(y0 - pad * h)), min(H, int(y1 + pad * h))
    return x0, y0, x1, y1


class YoloPose:
    name = "yolo11-pose"

    def __init__(self, size="s"):
        from ultralytics import YOLO
        self.m = YOLO(str(MODELS / f"yolo11{size}-pose.pt"))
        self.name = f"yolo11{size}-pose"

    def __call__(self, frame, box):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = crop_box(box, W, H, 0.4)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        r = self.m.predict(crop, imgsz=320, conf=0.15, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        k = r.keypoints.data[i].cpu().numpy().copy()
        k[:, 0] += x0
        k[:, 1] += y0
        return k


class VitPose:
    name = "vitpose-b"

    def __init__(self, model_id="usyd-community/vitpose-base-simple"):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = VitPoseForPoseEstimation.from_pretrained(model_id).eval().to(self.dev)

    def __call__(self, frame, box):
        from PIL import Image
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x0, y0, x1, y1 = box
        coco_box = [[[x0, y0, x1 - x0, y1 - y0]]]
        inp = self.proc(im, boxes=coco_box, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.m(**inp)
        res = self.proc.post_process_pose_estimation(out, boxes=coco_box)[0][0]
        k = res["keypoints"].cpu().numpy()
        s = res["scores"].cpu().numpy()
        return np.concatenate([k, s[:, None]], 1)


class RtmPose:
    name = "rtmpose-m"

    def __init__(self):
        from rtmlib import RTMPose
        onnx = next(MODELS.glob("rtm/**/rtmpose-m*/end2end.onnx"))
        import onnxruntime as ort
        dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        self.m = RTMPose(str(onnx), model_input_size=(192, 256), backend="onnxruntime", device=dev)

    def __call__(self, frame, box):
        kpts, scores = self.m(frame, bboxes=[list(box)])
        return np.concatenate([kpts[0], scores[0][:, None]], 1)


class MediaPipePose:
    name = "mediapipe-heavy"
    # MediaPipe 33 landmark index for each COCO-17 point
    MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, variant="heavy"):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        self.mp = mp
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / f"pose_landmarker_{variant}.task")),
            running_mode=RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.2)
        self.m = PoseLandmarker.create_from_options(opts)
        self.name = f"mediapipe-{variant}"

    def __call__(self, frame, box):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = crop_box(box, W, H, 0.4)
        crop = np.ascontiguousarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        if crop.size == 0:
            return None
        # MediaPipe needs a reasonably large person: upscale small crops
        scale = max(1.0, 256 / crop.shape[0])
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res = self.m.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        ch, cw = crop.shape[:2]
        k = np.array([[lm[i].x * cw / scale + x0, lm[i].y * ch / scale + y0, lm[i].visibility]
                      for i in self.MP2COCO], dtype=np.float32)
        return k


def build(name: str):
    name = name.lower()
    if name.startswith("yolo"):
        return YoloPose(name.split("-")[-1] if "-" in name else "s")
    if name == "vitpose":
        return VitPose()
    if name == "rtmpose":
        return RtmPose()
    if name == "mediapipe":
        return MediaPipePose()
    raise ValueError(name)


def draw_skeleton(img, k, thr=0.3, thickness=2):
    """Draw head (circle), torso, arms and legs as plain lines."""
    if k is None:
        return img
    pt = lambda i: (int(k[i, 0]), int(k[i, 1]))
    ok = lambda i: k[i, 2] >= thr
    for part, pairs in LIMBS.items():
        for a, b in pairs:
            if ok(a) and ok(b):
                cv2.line(img, pt(a), pt(b), COLORS[part], thickness, cv2.LINE_AA)
    # head: circle around the visible face points, joined to the neck
    face = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if ok(i)]
    if face:
        c = k[face, :2].mean(0)
        sh = [i for i in (LSH, RSH) if ok(i)]
        r = 6
        if len(sh) == 2:
            r = max(4, int(0.25 * np.linalg.norm(k[LSH, :2] - k[RSH, :2])))
            neck = k[[LSH, RSH], :2].mean(0)
            cv2.line(img, tuple(int(v) for v in c), tuple(int(v) for v in neck), COLORS["head"], thickness, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in c), r, COLORS["head"], thickness, cv2.LINE_AA)
    for i in range(5, 17):
        if ok(i):
            cv2.circle(img, pt(i), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img
