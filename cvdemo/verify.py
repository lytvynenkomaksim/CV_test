"""Untrained second check for buoy candidates: CLIP (general image-text model, no training) looks at an
enlarged crop around each candidate and says how much it looks like a buoy on water versus the usual
false alarms (spray, a person / ski, text / logo, plain water)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
POS = ["a small round buoy floating on water", "a colorful ball floating on a lake"]
NEG = ["white water spray", "a person water skiing", "a water ski", "text and a logo on a tv screen",
       "calm lake water", "a tree on the shore", "a boat"]


class ClipVerifier:
    name = "CLIP ViT-B/32 crop check (no training)"

    def __init__(self, pad=2.5):
        import clip
        import torch
        self.torch = torch
        self.m, self.pre = clip.load(str(MODELS / "clip/ViT-B-32.pt"), device="cpu")
        with torch.no_grad():
            t = self.m.encode_text(clip.tokenize(POS + NEG))
        self.t = t / t.norm(dim=-1, keepdim=True)
        self.npos, self.pad = len(POS), pad

    def __call__(self, frame, boxes):
        from PIL import Image
        H, W = frame.shape[:2]
        crops = []
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(24, self.pad * max(b[2] - b[0], b[3] - b[1]))
            x0, y0, x1, y1 = int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s))
            c = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            crops.append(self.pre(Image.fromarray(c)))
        with self.torch.no_grad():
            f = self.m.encode_image(self.torch.stack(crops))
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100 * f @ self.t.T).softmax(-1).numpy()
        return p[:, :self.npos].sum(1)
