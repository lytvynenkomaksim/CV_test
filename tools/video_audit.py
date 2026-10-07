#!/usr/bin/env python3
"""Audit water-ski videos: technical metadata + simple quality metrics.

For each video under --root (recursive) computes:
  resolution, fps, duration, frame count, bitrate, codec (ffprobe),
  sharpness   - mean variance-of-Laplacian of grayscale frames sampled at
                --sample-fps, resized to 640 px wide (comparable across videos),
  shake_px    - median global translation between consecutive sampled frames
                (cv2.phaseCorrelate, px at 640 width). Includes intentional
                camera panning, so treat it as "camera motion", not pure jitter,
  quality     - low / medium / high from resolution + sharpness,
  skier_ratio - fraction of sampled frames where YOLO detects a person with
                conf > --conf.

Writes <out>/video_audit.csv and <out>/video_audit.md.

Usage:
  python3 tools/video_audit.py                     # defaults: data/videos -> data/audit
  python3 tools/video_audit.py --root data/videos --out data/audit --threads 2
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}
WORK_W = 640

# quality tier thresholds (sharpness = var(Laplacian) at 640 px width).
# Calibrated on the 2026-10 test set: water spray gives high Laplacian variance,
# typical broadcast 600p clips score ~190-260, the 1080p Italy clip ~800.
SHARP_LOW = 150.0
SHARP_HIGH = 400.0


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0.0
    dur = float(info["format"].get("duration") or v.get("duration") or 0)
    nb = v.get("nb_frames")
    nframes = int(nb) if nb and nb.isdigit() else int(round(dur * fps))
    br = int(info["format"].get("bit_rate") or 0)
    return dict(width=int(v["width"]), height=int(v["height"]), fps=round(fps, 2),
                duration_s=round(dur, 2), frames=nframes,
                bitrate_kbps=round(br / 1000), codec=v.get("codec_name", "?"),
                size_mb=round(Path(path).stat().st_size / 1e6, 1))


def sample_frames(path, sample_fps):
    """Yield frames resized to WORK_W, sampled at ~sample_fps (sequential decode)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1.0, fps / sample_fps)
    nxt, idx = 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx >= nxt:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (WORK_W, int(round(h * WORK_W / w))),
                                   interpolation=cv2.INTER_AREA)
                yield frame
            nxt += step
        idx += 1
    cap.release()


def tier(meta, sharp):
    short = min(meta["width"], meta["height"])
    if short < 560 or sharp < SHARP_LOW:
        return "low"
    if short >= 900 and sharp >= SHARP_HIGH:
        return "high"
    return "medium"


def audit(path, model, sample_fps, conf):
    meta = ffprobe(path)
    sharps, shifts, person = [], [], 0
    prev, win, n = None, None, 0
    for frame in sample_frames(path, sample_fps):
        n += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharps.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        g32 = gray.astype(np.float32)
        if win is None or win.shape != g32.shape:
            win = cv2.createHanningWindow(g32.shape[::-1], cv2.CV_32F)
        if prev is not None and prev.shape == g32.shape:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g32, win)
            shifts.append(math.hypot(dx, dy))
        prev = g32
        if model is not None:
            r = model.predict(frame, classes=[0], conf=conf, verbose=False, imgsz=640)[0]
            if len(r.boxes):
                person += 1
    sharp = float(np.mean(sharps)) if sharps else 0.0
    row = dict(file=str(path), **meta,
               sampled=n,
               sharpness=round(sharp, 1),
               shake_px=round(float(np.median(shifts)), 2) if shifts else 0.0,
               quality=tier(meta, sharp),
               skier_ratio=round(person / n, 3) if (n and model is not None) else "")
    return row


def write_md(rows, path, root):
    cols = ["video", "res", "fps", "dur_s", "frames", "kbps", "codec", "MB",
            "sharpness", "shake_px", "quality", "skier_ratio"]
    lines = ["# Video audit", "",
             f"Sharpness = mean var(Laplacian) on grayscale frames at {WORK_W}px width; "
             "shake_px = median phase-correlation shift between consecutive sampled "
             f"frames (px at {WORK_W}px width, includes intentional panning); "
             "skier_ratio = share of sampled frames with a YOLO person (conf>thr). "
             f"Quality tier: low if short side <560 or sharpness <{SHARP_LOW:g}; "
             f"high if short side >=900 and sharpness >={SHARP_HIGH:g}; else medium.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        try:
            name = str(Path(r["file"]).relative_to(root))
        except ValueError:
            name = r["file"]
        vals = [name, f'{r["width"]}x{r["height"]}', r["fps"], r["duration_s"],
                r["frames"], r["bitrate_kbps"], r["codec"], r["size_mb"],
                r["sharpness"], r["shake_px"], r["quality"], r["skier_ratio"]]
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in vals) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "data" / "videos"))
    ap.add_argument("--out", default=str(REPO / "data" / "audit"))
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--model", default=None,
                    help="YOLO weights (default: models/yolo11n.pt if present, else yolo11n.pt)")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    cv2.setNumThreads(args.threads)
    model = None
    if not args.no_yolo:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(args.threads)
        w = args.model or (str(REPO / "models" / "yolo11n.pt")
                           if (REPO / "models" / "yolo11n.pt").exists() else "yolo11n.pt")
        model = YOLO(w)

    root = Path(args.root).resolve()
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    rows = []
    for i, p in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}] {p.relative_to(root)}", file=sys.stderr, flush=True)
        try:
            row = audit(p, model, args.sample_fps, args.conf)
        except Exception as e:  # keep going on broken files
            print(f"  failed: {e}", file=sys.stderr)
            continue
        row["file"] = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "video_audit.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        write_md(rows, out / "video_audit.md", REPO)
    print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
