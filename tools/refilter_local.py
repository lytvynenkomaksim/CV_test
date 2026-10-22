"""Re-check buoys found by local search / gap filling in saved results against the video frames: keep a box
only if it stands out in colour from the water around it (cvdemo.temporal.stands_out). For results made
before that check was part of the tracker.

    python tools/refilter_local.py results/hq_full
"""
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cvdemo.temporal import stands_out  # noqa: E402

d = Path(sys.argv[1])
for js in sorted(d.glob("*.frames.json")):
    tag = js.name[:-len(".frames.json")]
    meta = json.loads((d / f"{tag}.json").read_text())["meta"]
    frames = json.loads(js.read_text())
    cap = cv2.VideoCapture(str(ROOT / meta["video"].split("CV_test/")[-1]))
    kept = dropped = 0
    for rec in tqdm(frames, desc=tag[:30]):
        ok, img = cap.read()
        if not ok or not rec.get("buoys"):
            continue
        how = rec.get("buoy_how") or ["det"] * len(rec["buoys"])
        nb, nh = [], []
        for b, h in zip(rec["buoys"], how):
            if h == "det" or stands_out(img, b):
                nb.append(b)
                nh.append(h)
                kept += 1
            else:
                dropped += 1
        rec["buoys"], rec["buoy_how"] = nb, nh
    js.write_text(json.dumps(frames))
    print(tag, "kept", kept, "dropped", dropped)
