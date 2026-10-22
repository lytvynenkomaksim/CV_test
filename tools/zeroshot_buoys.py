"""Buoy detection WITHOUT training: run several off-the-shelf detectors only inside the zone near the skier.

    python tools/zeroshot_buoys.py --gold data/buoy_gold --out results/zeroshot [--models hsv,yoloe26l,...]

Each detector sees the ROI crop (the water band around the skier's depth, see tools/buoy_gold_sample.py),
upscaled to --crop-w px wide, so tiny far buoys get more pixels. Predictions (all scores >= 0.03) are saved
to <out>/preds/<model>.json in full-frame coordinates; scoring against the hand-checked boxes is done by
tools/zeroshot_score.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
PROMPTS = ["buoy", "orange ball", "red ball", "yellow ball"]

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=str(ROOT / "data/buoy_gold"))
ap.add_argument("--out", default=str(ROOT / "results/zeroshot"))
ap.add_argument("--models", default="hsv,coco_ball,yoloworld_s,yoloworld_m,yoloe11l,yoloe26l,yoloe26l_visual,"
                "gdino_tiny,gdino_base,mmgdino_tiny,owlv2,owlv2_image,finetuned")
ap.add_argument("--crop-w", type=int, default=1280)
ap.add_argument("--full", action="store_true", help="run on the full frame instead of the ROI crop")
a = ap.parse_args()


# ---------------------------------------------------------------- detectors: img (BGR crop) -> [(x0,y0,x1,y1,score)]
def hsv_detector():
    """Classical: buoys are small, strongly coloured blobs (orange, red, pink, yellow, green) on the water.
    Anything saturated whose hue differs from this frame's water hue counts; white buoys are NOT found
    (white = spray)."""
    def run(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        water_h = float(np.median(h[s > 40])) if (s > 40).any() else 90.0
        dh = np.minimum(np.abs(h.astype(int) - water_h), 180 - np.abs(h.astype(int) - water_h))
        warm = (dh >= 18) & (s >= 110) & (v >= 90)
        m = cv2.morphologyEx(warm.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            x, y, w, hh, area = st[i]
            if area < 6 or w > 0.15 * img.shape[1] or not (0.35 < w / max(hh, 1) < 3.5):
                continue
            fill = area / float(w * hh)
            sat = float(s[lab == i].mean()) / 255
            out.append((x, y, x + w, y + hh, round(min(1.0, 0.5 * sat + 0.5 * fill), 3)))
        return out
    return run


def ultra(weights, classes=None, coco_ids=None, imgsz=1280):
    from ultralytics import YOLO, YOLOE
    m = (YOLOE if "yoloe" in weights else YOLO)(str(MODELS / weights))
    if classes:
        if "yoloe" in weights:
            m.set_classes(classes, m.get_text_pe(classes))
        else:
            m.set_classes(classes)

    def run(img):
        r = m.predict(img, imgsz=imgsz, conf=0.03, classes=coco_ids, verbose=False)[0]
        return [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    return run


def yoloe_visual(weights, ref):
    """YOLOE visual prompt: show it example buoys in ONE reference frame (from a different video), no text,
    no training."""
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    m = YOLOE(str(MODELS / weights))
    ref_img = cv2.imread(str(ref["image"]))
    vp = dict(bboxes=np.array(ref["boxes"], float), cls=np.zeros(len(ref["boxes"]), int))

    def run(img):
        r = m.predict(img, refer_image=ref_img, visual_prompts=vp, predictor=YOLOEVPSegPredictor, imgsz=1280,
                      conf=0.03, verbose=False)[0]
        return [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
    return run


def hf_grounding(repo):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(repo, cache_dir=MODELS / "hf/hub")
    m = AutoModelForZeroShotObjectDetection.from_pretrained(repo, cache_dir=MODELS / "hf/hub").eval()
    text = " . ".join(PROMPTS) + " ."

    def run(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = proc(images=rgb, text=text, return_tensors="pt")
        with torch.no_grad():
            o = m(**inp)
        r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=0.1, text_threshold=0.1,
                                                        target_sizes=[rgb.shape[:2]])[0]
        return [(*b, s) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]
    return run


def owlv2(image_ref=None):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    repo = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(repo, cache_dir=MODELS / "hf/hub")
    m = Owlv2ForObjectDetection.from_pretrained(repo, cache_dir=MODELS / "hf/hub").eval()
    query = None
    if image_ref is not None:  # image-guided: embed one example buoy crop
        ri = cv2.imread(str(image_ref["image"]))
        x0, y0, x1, y1 = map(int, image_ref["boxes"][0])
        pad = max(4, int(0.6 * (x1 - x0)))
        q = ri[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        query = Image.fromarray(cv2.cvtColor(q, cv2.COLOR_BGR2RGB))

    def run(img):
        H, W = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        S = max(H, W)  # OWLv2 pads to a square: boxes are relative to the padded size
        with torch.no_grad():
            if query is None:
                inp = proc(text=[[f"a photo of a {p}" for p in PROMPTS]], images=pil, return_tensors="pt")
                o = m(**inp)
                r = proc.post_process_grounded_object_detection(o, threshold=0.03, target_sizes=[(S, S)])[0]
            else:
                inp = proc(images=pil, query_images=query, return_tensors="pt")
                o = m.image_guided_detection(**inp)
                r = proc.post_process_image_guided_detection(o, threshold=0.5, nms_threshold=0.3,
                                                             target_sizes=[(S, S)])[0]
        return [(*b, s) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]
    return run


def colorblob(verify=False):
    from cvdemo.colorblob import ColorBlobBuoy
    from cvdemo.verify import ClipVerifier
    d = ColorBlobBuoy(verifier=ClipVerifier() if verify else None)

    def run(img):  # img = the zone crop already
        c = [b[:5] for b in d.candidates(img)]
        if verify and c:
            c = sorted(c, key=lambda b: -b[4])[:d.max_verify]
            c = [(*b[:4], float(p)) for b, p in zip(c, d.verifier(img, c))]
        return c
    return run


def build(name, ref):
    if name.startswith("yolo="):  # any trained buoy model: yolo=<weights under models/>[@imgsz]
        w, _, sz = name[5:].partition("@")
        return ultra(w, imgsz=int(sz or 1280))
    return {
        "hsv": hsv_detector,
        "colorblob": lambda: colorblob(False),
        "colorblob_clip": lambda: colorblob(True),
        "coco_ball": lambda: ultra("yolo11m.pt", coco_ids=[32]),
        "yoloworld_s": lambda: ultra("yolov8s-worldv2.pt", PROMPTS),
        "yoloworld_m": lambda: ultra("yolov8m-worldv2.pt", PROMPTS),
        "yoloe11l": lambda: ultra("yoloe-11l-seg.pt", PROMPTS),
        "yoloe26l": lambda: ultra("yoloe-26l-seg.pt", PROMPTS),
        "yoloe26l_visual": lambda: yoloe_visual("yoloe-26l-seg.pt", ref),
        "gdino_tiny": lambda: hf_grounding("IDEA-Research/grounding-dino-tiny"),
        "gdino_base": lambda: hf_grounding("IDEA-Research/grounding-dino-base"),
        "mmgdino_tiny": lambda: hf_grounding("openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"),
        "owlv2": lambda: owlv2(),
        "owlv2_image": lambda: owlv2(ref),
        "finetuned": lambda: ultra("../weights/buoy_yolo11s.pt" if (ROOT / "weights/buoy_yolo11s.pt").exists()
                                   else "buoy_yolo11s.pt"),
    }[name]()


if __name__ == "__main__":
    gold = Path(a.gold)
    items = json.loads((gold / "items.json").read_text())
    labels = json.loads((gold / "labels.json").read_text()) if (gold / "labels.json").exists() else {}
    # reference example for the visual / image-guided prompts: a clear labelled buoy from a held-out video
    ref = json.loads((gold / "reference.json").read_text()) if (gold / "reference.json").exists() else None
    if ref:
        ref["image"] = gold / ref["image"]
    out = Path(a.out) / ("preds_full" if a.full else "preds")
    out.mkdir(parents=True, exist_ok=True)
    timing = {}
    for name in a.models.split(","):
        if "visual" in name or "image" in name:
            if not ref:
                print("skip", name, "(no reference.json)")
                continue
        det = build(name, ref)
        preds, ts = {}, []
        for it in tqdm(items, desc=name):
            if ref and it["video"] == ref.get("video"):  # never test on the reference video
                continue
            img = cv2.imread(str(gold / it["image"]))
            x0, y0, x1, y1 = it["roi"] if not a.full else (0, 0, it["W"], it["H"])
            crop = img[y0:y1, x0:x1]
            sc = a.crop_w / crop.shape[1] if not a.full else 1.0
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc != 1 else crop
            t0 = time.perf_counter()
            boxes = det(crop)
            ts.append(time.perf_counter() - t0)
            preds[it["image"]] = [[b[0] / sc + x0, b[1] / sc + y0, b[2] / sc + x0, b[3] / sc + y0, float(b[4])]
                                  for b in boxes]
        name = name.replace("yolo=", "").replace(".pt", "").replace("@", "_")
        (out / f"{name}.json").write_text(json.dumps(preds))
        timing[name] = round(1000 * float(np.median(ts[1:] or ts)), 1)
        print(name, "ms/crop", timing[name])
        tf = out / "timing.json"
        old = json.loads(tf.read_text()) if tf.exists() else {}
        old.update(timing)
        tf.write_text(json.dumps(old, indent=1))
