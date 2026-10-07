"""End-to-end demo: skier detection + tracking, skeleton, buoy detection + tracking,
buoy-pass events and an annotated output video.

    python -m cvdemo.pipeline VIDEO --buoy yolo-buoy --pose yolo-s --out runs/demo

Passes:
  1. analyse  - run the models frame by frame, store per-frame results (+ timings)
  2. reason   - track buoys in pan-compensated coords, find crossings, judge passes
  3. render   - draw everything on the video (H.264 mp4)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import buoys as B
from . import geometry as G
from . import pose as P

MODELS = Path(__file__).resolve().parent.parent / "models"


class Timer:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def __call__(self, key):
        timer = self

        class _C:
            def __enter__(s):
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                timer.t[key] += time.perf_counter() - s.t0
                timer.n[key] += 1
        return _C()

    def summary(self):
        return {k: dict(total_s=round(v, 2), calls=self.n[k], ms_per_call=round(1000 * v / max(1, self.n[k]), 1))
                for k, v in self.t.items()}


class SkierTracker:
    """YOLO11 person detector + ByteTrack; picks the skier (most persistent, confident track)."""

    def __init__(self, size="s", conf=0.2, imgsz=960):
        from ultralytics import YOLO
        self.m = YOLO(str(MODELS / f"yolo11{size}.pt"))
        self.name = f"yolo11{size} + ByteTrack"
        self.conf, self.imgsz = conf, imgsz
        self.score = defaultdict(float)
        self.last = None

    def reset(self):
        # new shot -> fresh tracker state
        if getattr(self.m, "predictor", None) is not None and hasattr(self.m.predictor, "trackers"):
            for t in self.m.predictor.trackers:
                t.reset()
        self.last = None

    def __call__(self, frame):
        r = self.m.track(frame, persist=True, classes=[0], conf=self.conf, imgsz=self.imgsz,
                         tracker="bytetrack.yaml", verbose=False)[0]
        boxes = r.boxes.xyxy.tolist()
        confs = r.boxes.conf.tolist()
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(boxes)
        H, W = frame.shape[:2]
        best, bs = None, -1
        for b, c, i in zip(boxes, confs, ids):
            h = b[3] - b[1]
            if b[3] < 0.25 * H or h > 0.8 * H:  # skier is on the water, never the full frame
                continue
            if i >= 0:
                self.score[i] += c
            s = c + 0.02 * self.score.get(i, 0)
            if self.last is not None:  # temporal continuity
                d = np.hypot((b[0] + b[2]) / 2 - self.last[0], b[3] - self.last[1])
                s -= d / (2 * W)
            if s > bs:
                best, bs = (b, c, i), s
        if best:
            b = best[0]
            self.last = ((b[0] + b[2]) / 2, b[3])
        persons = [b for b, c in zip(boxes, confs) if c > 0.3]
        return best, persons


def analyse(video, buoy_det, pose_est, skier, buoy_every=1, max_frames=None, timer=None, log_every=50):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pan_est, cuts = G.PanEstimator(), G.CutDetector()
    frames, shots = [], [0]
    timer = timer or Timer()
    f = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        with timer("scene_cut"):
            if cuts(frame):
                shots.append(f)
                pan_est.reset()
                skier.reset()
        with timer("camera_pan"):
            pan = pan_est.update(frame)
        with timer("skier_detect_track"):
            best, persons = skier(frame)
        rec = dict(pan=pan, skier=None, kpts=None, buoys=None, skier_X=None, foot_y=None)
        if best:
            b, c, tid = best
            rec.update(skier=[*map(float, b), float(c), int(tid)], skier_X=(b[0] + b[2]) / 2 + pan, foot_y=b[3])
            if pose_est is not None:
                with timer("pose"):
                    k = pose_est(frame, b)
                if k is not None:
                    rec["kpts"] = np.round(k, 2).tolist()
        if buoy_det is not None and f % buoy_every == 0:
            with timer("buoy_detect"):
                rec["buoys"] = [list(map(float, d)) for d in buoy_det(frame, persons)]
        frames.append(rec)
        f += 1
        if log_every and f % log_every == 0:
            el = time.perf_counter() - t_start
            print(f"  frame {f}/{n}  {f / el:.2f} fps", flush=True)
    cap.release()
    meta = dict(video=str(video), fps=fps, frames=len(frames), width=W, height=H, shots=shots,
                wall_s=round(time.perf_counter() - t_start, 2))
    return frames, meta


def reason(frames, meta, buoy_every=1):
    shots = meta["shots"] + [len(frames)]
    all_events, seg_info, tracks_all = [], [], []
    for s0, s1 in zip(shots[:-1], shots[1:]):
        tr = B.BuoyTracker(max_miss=max(15, 3 * buoy_every))
        for f in range(s0, s1):
            r = frames[f]
            if r["buoys"] is None:
                continue
            tr.update(f, r["buoys"], r["pan"])
        tracks = tr.all_tracks()
        ev, C, A = G.buoy_events(tracks, frames, s0, s1)
        all_events += ev
        seg_info.append(dict(start=s0, end=s1, centre=C, amp=A))
        tracks_all += [(s0, s1, t) for t in tracks]
    return all_events, seg_info, tracks_all


def render(video, out_path, frames, meta, events, segs, tracks, labels, timings=None):
    cap = cv2.VideoCapture(str(video))
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    tmp = str(out_path) + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    # per-frame buoy draw list from tracks (observed + coasting)
    draw = defaultdict(list)
    ev_by_track = {e["track"]: e for e in events}
    for s0, s1, t in tracks:
        if t.n_obs() < 3:
            continue
        for h in t.hist:
            draw[h[0]].append((t.tid, h))
    seg_of = lambda f: next((s for s in segs if s["start"] <= f < s["end"]), None)
    turn_events = [e for e in events if e["kind"] == "turn"]
    trail = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok or f >= len(frames):
            break
        r = frames[f]
        pan = r["pan"]
        seg = seg_of(f)
        if f in meta["shots"]:
            trail = []
        # centre line
        if seg and seg["centre"] is not None:
            cx = int(seg["centre"] - pan)
            if 0 <= cx < W:
                for y in range(int(H * 0.3), H, 14):
                    cv2.line(img, (cx, y), (cx, y + 7), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, "course centre", (cx + 4, int(H * 0.3) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
        # buoys
        for tid, h in draw.get(f, []):
            _, X, y, w, hh, s, obs = h
            x = X - pan
            ev = ev_by_track.get(tid)
            col = (0, 165, 255)
            tag = f"b{tid}"
            if ev:
                if ev["kind"] == "turn":
                    col = (0, 200, 0) if ev["outside"] else (0, 0, 255)
                    tag = f"TURN {ev['side'][0].upper()} #{tid}"
                else:
                    col, tag = (0, 255, 255), f"centre #{tid}"
            x0, y0, x1, y1 = int(x - w / 2) - 3, int(y - hh) - 3, int(x + w / 2) + 3, int(y) + 3
            if obs:
                cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            else:  # predicted (hidden by spray / missed detection)
                cv2.circle(img, (int(x), int(y - hh / 2)), int(max(w, hh)) + 4, col, 1, cv2.LINE_AA)
            cv2.putText(img, tag, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        # skier + skeleton + trail
        if r["skier"]:
            x0, y0, x1, y1, c, tid = r["skier"]
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1)
            cv2.putText(img, f"skier id{tid} {c:.2f}", (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 0, 255), 1, cv2.LINE_AA)
            trail.append((r["skier_X"], r["foot_y"]))
            if r["kpts"] is not None:
                P.draw_skeleton(img, np.array(r["kpts"]))
        trail = trail[-int(fps * 2):]
        pts = [(int(X - pan), int(y)) for X, y in trail]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (255, 0, 255), 1, cv2.LINE_AA)
        # crossing flashes
        for e in events:
            if 0 <= f - e["frame"] < int(fps * 1.2):
                bx, sx = int(e["buoy_X"] - pan), int(e["skier_X"] - pan)
                y = int(r["foot_y"]) if r["foot_y"] else H // 2
                if e["kind"] == "turn":
                    ok_ = e["outside"]
                    col = (0, 200, 0) if ok_ else (0, 0, 255)
                    msg = f"{e['side'].upper()} buoy: skier {'OUTSIDE - OK' if ok_ else 'INSIDE - MISS'}"
                    if e["predicted"]:
                        msg += " (buoy position predicted)"
                else:
                    col, msg = (0, 255, 255), "centre buoy (gate/guide) passed"
                if f - e["frame"] < 3:
                    cv2.line(img, (bx, y), (sx, y), col, 2, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (W // 2 - tw // 2 - 8, 60), (W // 2 + tw // 2 + 8, 92), (0, 0, 0), -1)
                cv2.putText(img, msg, (W // 2 - tw // 2, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        # HUD (bottom-left)
        done = [e for e in turn_events if e["frame"] <= f]
        lines = [f"models: skier={labels['skier']} | pose={labels['pose']} | buoy={labels['buoy']}",
                 "turn buoys: " + " ".join(("OK" if e["outside"] else "X") + e["side"][0].upper() for e in done)
                 + f"   score {G.score_pass(done)['score']}"]
        if timings:
            lines.append(timings)
        y = H - 12 - 18 * (len(lines) - 1)
        ov = img.copy()
        cv2.rectangle(ov, (0, y - 16), (W, H), (0, 0, 0), -1)
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        for i, l in enumerate(lines):
            cv2.putText(img, l, (8, y + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
        f += 1
    vw.release()
    cap.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], check=True)
    Path(tmp).unlink()


def skier_stats(frames, meta):
    vis = [r["skier"] is not None for r in frames]
    pose_ok = [r["kpts"] is not None and np.mean(np.array(r["kpts"])[:, 2] > 0.3) > 0.5 for r in frames]
    hs = [r["skier"][3] - r["skier"][1] for r in frames if r["skier"]]
    return dict(skier_visible_ratio=round(float(np.mean(vis)), 3) if vis else 0,
                skier_visible_s=round(sum(vis) / meta["fps"], 2),
                pose_good_ratio=round(float(np.mean(pose_ok)), 3) if pose_ok else 0,
                skier_median_height_px=round(float(np.median(hs)), 1) if hs else None)


def buoy_stats(frames, tracks, meta):
    """How often buoys are visible and how often a tracked buoy is hidden (spray / missed)."""
    det_frames = [r for r in frames if r["buoys"] is not None]
    with_buoy = [len(r["buoys"]) > 0 for r in det_frames]
    good = [t for _, _, t in tracks if t.n_obs() >= 3]
    hidden, total = 0, 0
    gaps = []
    for t in good:
        obs_f = [h[0] for h in t.hist if h[6]]
        span = obs_f[-1] - obs_f[0] + 1
        total += span
        hidden += span - len(obs_f)
        g = np.diff(obs_f) - 1
        gaps += [int(x) for x in g if x > 0]
    return dict(frames_with_buoy_ratio=round(float(np.mean(with_buoy)), 3) if with_buoy else 0,
                buoy_tracks=len(good),
                hidden_ratio_within_tracks=round(hidden / total, 3) if total else None,
                longest_hidden_gap_s=round(max(gaps) / meta["fps"], 2) if gaps else 0)


def run(video, out_dir, buoy="yolo-buoy", pose="yolo-s", skier_size="s", buoy_every=1, max_frames=None,
        tag=None, buoy_kw=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or Path(video).stem.replace(" ", "_")
    timer = Timer()
    with timer("load_models"):
        skier = SkierTracker(skier_size)
        pose_est = P.build(pose) if pose and pose != "none" else None
        buoy_det = B.build(buoy, **(buoy_kw or {})) if buoy and buoy != "none" else None
    labels = dict(skier=skier.name, pose=pose_est.name if pose_est else "-", buoy=buoy_det.name if buoy_det else "-")
    print(f"[{tag}] analysing with {labels}", flush=True)
    frames, meta = analyse(video, buoy_det, pose_est, skier, buoy_every, max_frames, timer)
    events, segs, tracks = reason(frames, meta, buoy_every)
    score = G.score_pass(events)
    tsum = timer.summary()
    proc_s = sum(v["total_s"] for k, v in tsum.items() if k != "load_models")
    video_s = meta["frames"] / meta["fps"]
    timing_line = (f"CPU processing {proc_s / meta['frames'] * 1000:.0f} ms/frame "
                   f"({proc_s / video_s:.1f}x real-time), buoy detector every {buoy_every} frame(s)")
    out_mp4 = out_dir / f"{tag}.mp4"
    render(video, out_mp4, frames, meta, events, segs, tracks, labels, timing_line)
    result = dict(meta=meta, models=labels, buoy_every=buoy_every, timings=tsum,
                  processing_s=round(proc_s, 2), video_s=round(video_s, 2),
                  x_realtime=round(proc_s / video_s, 2), skier=skier_stats(frames, meta),
                  buoys=buoy_stats(frames, tracks, meta), events=events, score=score, segments=segs,
                  output_video=str(out_mp4))
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float))
    (out_dir / f"{tag}.frames.json").write_text(json.dumps(frames, default=float))
    print(f"[{tag}] done: {score}  {timing_line}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--buoy", default="yolo-buoy", choices=["yolo-buoy", "gdino", "yoloworld", "owlv2", "none"])
    ap.add_argument("--pose", default="yolo-s", help="yolo-n|yolo-s|yolo-m|vitpose|rtmpose|mediapipe|none")
    ap.add_argument("--skier", default="s", help="YOLO11 size for the person detector (n/s/m)")
    ap.add_argument("--buoy-every", type=int, default=1, help="run the buoy detector every N frames")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    run(a.video, a.out, a.buoy, a.pose, a.skier, a.buoy_every, a.max_frames, a.tag)


if __name__ == "__main__":
    main()
