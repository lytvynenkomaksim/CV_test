"""Auto-label buoys (and skier/boat for reference) with Grounding DINO.

Samples frames from videos at a fixed rate, runs zero-shot Grounding DINO with the
prompt "buoy. person. boat." and writes a YOLO-format dataset (class 0 = buoy)
that is used to train a fast, specialised buoy detector (see train_buoy_yolo.py).

Usage: python tools/autolabel_gdino.py --videos data/videos/front_view/*.mp4 \
          --out data/buoy_dataset --every 0.5 [--holdout NAME ...]
"""
import argparse, json, os, time
from pathlib import Path
import cv2, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def iou_contained(b, p):
    """Fraction of box b covered by box p."""
    x0, y0 = max(b[0], p[0]), max(b[1], p[1])
    x1, y1 = min(b[2], p[2]), min(b[3], p[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="data/buoy_dataset")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between samples")
    ap.add_argument("--thr", type=float, default=0.28)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--holdout", nargs="*", default=[], help="substrings of videos to put in val split")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).eval().to(dev)
    out = Path(a.out)
    meta = []
    for vp in a.videos:
        split = "val" if any(h in vp for h in a.holdout) else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(fps * a.every))
        stem = Path(vp).stem.replace(" ", "_").replace("(", "").replace(")", "")
        for fi in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t = time.time()
            inp = proc(images=im, text="buoy. person. boat.", return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**inp)
            r = proc.post_process_grounded_object_detection(
                o, inp.input_ids, threshold=0.15, text_threshold=0.15, target_sizes=[(H, W)])[0]
            labels = r.get("text_labels", r["labels"])
            dets = [(l, float(s), [float(v) for v in b]) for l, s, b in zip(labels, r["scores"], r["boxes"])]
            persons = [b for l, s, b in dets if l == "person" and s > 0.3]
            buoys = []
            for l, s, b in dets:
                if l != "buoy" or s < a.thr:
                    continue
                w, h = b[2] - b[0], b[3] - b[1]
                if w > 0.06 * W or h > 0.08 * H or w * h < 9:  # buoys are small
                    continue
                if any(iou_contained(b, p) > 0.5 for p in persons):
                    continue
                buoys.append((s, b))
            name = f"{stem}_f{fi:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                for s, b in buoys:
                    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
                    f.write(f"0 {cx:.6f} {cy:.6f} {(b[2]-b[0])/W:.6f} {(b[3]-b[1])/H:.6f}\n")
            meta.append({"image": name, "split": split, "video": vp, "frame": fi, "sec": time.time() - t,
                         "buoys": [[round(s, 3)] + [round(v, 1) for v in b] for s, b in buoys],
                         "all": [[l, round(s, 3)] + [round(v, 1) for v in b] for l, s, b in dets]})
            print(f"{name} {split} buoys={len(buoys)} {time.time()-t:.1f}s", flush=True)
    (out / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n  0: buoy\n")
    with open(out / "autolabel_meta.jsonl", "a") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    main()
