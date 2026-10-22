"""Score buoy detectors against the hand-checked boxes, only inside the zone near the skier.

    python tools/zeroshot_score.py --gold data/buoy_gold --preds results/zeroshot/preds --out results/zeroshot
A prediction is a hit when its centre is within max(6 px, 0.75 x box size) of an unmatched true buoy
(IoU is too harsh for 6-15 px objects). Per model: AP, and P / R / F1 at the best-F1 score threshold
(that threshold is reported - in use it would be fixed per model), recall per zone (LEFT / MIDDLE / RIGHT
third of the image) and per size, false alarms per frame, ms per zone crop on this CPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default="data/buoy_gold")
ap.add_argument("--preds", default="results/zeroshot/preds")
ap.add_argument("--out", default="results/zeroshot")
ap.add_argument("--no-heldout", dest="heldout", action="store_false",
                help="skip the held-out table (all test videos are unseen by the fine-tuned model)")
a = ap.parse_args()
gold = Path(a.gold)
items = json.loads((gold / "items.json").read_text())
GT = json.loads((gold / "gold.json").read_text())  # frames not in gold.json (shore-camera shots) are not scored
IGN = json.loads((gold / "ignore.json").read_text()) if (gold / "ignore.json").exists() else {}
items = [it for it in items if it["image"] in GT]


def ignored(b, im):
    """Predictions on a spot the reviewer could not decide (too faint / too far) count neither way."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(g[0] - 4 <= cx <= g[2] + 4 and g[1] - 4 <= cy <= g[3] + 4 for g in IGN.get(im, []))
timing = json.loads((Path(a.preds) / "timing.json").read_text())


def in_roi(b, it):
    return it["roi"][1] <= (b[1] + b[3]) / 2 <= it["roi"][3]


def zone(b, it):
    cx = (b[0] + b[2]) / 2
    return ["LEFT", "MIDDLE", "RIGHT"][min(2, int(3 * cx / it["W"]))]


def evaluate(P, its):
    dets, n_gt = [], 0
    gt_meta = []
    for it in its:
        g = GT.get(it["image"], [])
        n_gt += len(g)
        gt_meta += [(it["image"], k, zone(b, it), b[3] - b[1]) for k, b in enumerate(g)]
        for b in P.get(it["image"], []):
            if in_roi(b, it) and not ignored(b, it["image"]):
                dets.append((b[4], it["image"], b))
    dets.sort(key=lambda d: -d[0])
    used, tp = set(), []
    for s, im, b in dets:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best = None
        for k, g in enumerate(GT.get(im, [])):
            gx, gy, gs = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2, max(g[2] - g[0], g[3] - g[1])
            d = np.hypot(cx - gx, cy - gy)
            if (im, k) not in used and d <= max(6, 0.75 * gs) and (best is None or d < best[0]):
                best = (d, k)
        if best:
            used.add((im, best[1]))
        tp.append((s, best is not None, (im, best[1]) if best else None))
    if not tp or not n_gt:
        return None
    hits = np.cumsum([t[1] for t in tp])
    prec = hits / np.arange(1, len(tp) + 1)
    rec = hits / n_gt
    ap_ = float(np.trapezoid(np.maximum.accumulate(prec[::-1])[::-1], rec)) if len(rec) > 1 else 0.0
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.argmax(f1))
    thr = tp[i][0]
    matched = {t[2] for t in tp[:i + 1] if t[1]}
    by_zone = {}
    for im, k, z, h in gt_meta:
        by_zone.setdefault(z, []).append((im, k) in matched)
    small = [((im, k) in matched) for im, k, z, h in gt_meta if h < 12]
    return dict(AP=round(ap_, 2), P=round(float(prec[i]), 2), R=round(float(rec[i]), 2), F1=round(float(f1[i]), 2),
                thr=round(float(thr), 2), fp_per_frame=round((i + 1 - hits[i]) / len(its), 2),
                **{f"R_{z}": f"{np.mean(v):.2f} ({len(v)})" for z, v in sorted(by_zone.items())},
                R_small=f"{np.mean(small):.2f} ({len(small)})" if small else "-")


# videos the fine-tuned buoy model never saw in training (its val split + the YouTube video)
HELD_OUT = ("Aaron_Davis", "Lucas Cornale 6 at 35", "Vincenzo Marino 6 at 35", "youtube")
md = []
groups = [("All test frames", lambda it: True)]
if a.heldout:
    groups.append(("Held-out videos only (fair for the fine-tuned model)",
                   lambda it: any(h in it["video"] for h in HELD_OUT)))
for title, keep in groups:
    rows = []
    for p in sorted(Path(a.preds).glob("*.json")):
        if p.stem == "timing":
            continue
        P = json.loads(p.read_text())
        its = [it for it in items if it["image"] in P and keep(it)]
        r = evaluate(P, its)
        if r:
            rows.append(dict(model=p.stem, frames=len(its), **r, ms_per_crop=timing.get(p.stem)))
    rows.sort(key=lambda r: -r["F1"])
    sel = [it for it in items if keep(it)]
    hdr = list(rows[0])
    md += [f"### {title}: {sum(len(GT.get(it['image'], [])) for it in sel)} hand-checked buoys, {len(sel)} frames, "
           f"{len({it['video'] for it in sel})} videos", "",
           "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows] + [""]
md = ["Buoy detection inside the zone near the skier only. A hit = predicted centre within max(6 px, 0.75 x size) "
      "of a true buoy. P/R/F1 at the best-F1 threshold (thr). R_small = buoys under 12 px tall. "
      "ms = per zone crop (1280 px wide) on 4 CPU cores.", ""] + md
Path(a.out, "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
