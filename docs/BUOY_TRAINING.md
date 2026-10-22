# Fast buoy detector: train it on your Mac

## Do we need to fine-tune at all?

The zero-shot test (`results/zeroshot/summary.md`) was scored against hand-checked buoys, only in the zone near
the skier:

- Without any training, YOLOE-26L with the text prompt "buoy" reaches F1 0.83 (0.89 on the 4 videos our
  fine-tuned model never saw).
- The fine-tuned model reaches 0.85 (0.93 on those 4 videos), and it is 5x faster.

So training is optional for accuracy. It is mainly worth it for speed and for removing specific confusions.
This training run checks whether a tiny zone-only model (YOLO11n) can match both.

## What it trains

- **Model:** YOLO11n (2.6 M parameters), one class `buoy`, 640 px input.
- **Training images:** only the zone near the skier. That is the water band from 1 skier-height beyond the ski
  to 4 heights closer to the boat, cut into LEFT / MIDDLE / RIGHT tiles.
- **Real tiles:** cleaned labels from `data/buoy_dataset_v2`.
  - Missed buoys are never taught as background. Anything a high-recall search finds that has no label
    (the old detector at confidence 0.05, plus colour blobs) is painted over with the surrounding water first.
  - The skier and ski are kept as true negatives.
- **More real buoys:** about 950 real buoy cut-outs, harvested from all frames of the training videos.
  - Only confident detections on buoy tracks that last at least 8 frames are used.
  - Overlay digits and dark patches are filtered out.
  - The cut-outs are pasted, in their real colours, onto a copy of every tile. Pasting avoids TV captions.
- **Never used for training:** Aaron Davis, Lucas Cornale 6@35, Vincenzo Marino 6@35 and the YouTube video.
  These are the test set.

## Run it (macOS, Apple silicon; also works on Linux / Windows)

```bash
git clone https://github.com/lytvynenkomaksim/CV_test.git && cd CV_test
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics tqdm
mkdir -p models && curl -L -o models/yolo11n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

python tools/extract_label_frames.py --dataset data/buoy_dataset_v2   # frames from the repo videos (~2 min)
python tools/make_zone_dataset.py                                     # tiles + real-buoy pastes (~10-15 min)
python tools/train_buoy_yolo.py --data data/buoy_zone/data.yaml --model yolo11n.pt --imgsz 640 \
    --epochs 30 --patience 8 --batch 16 --workers 4 --name buoy_zone_yolo11n --out buoy_zone_yolo11n.pt
python tools/zeroshot_buoys.py --models yolo=buoy_zone_yolo11n.pt@1280   # test on the hand-checked buoys
python tools/zeroshot_score.py                                            # -> results/zeroshot/summary.md
```

The training script picks the Apple GPU (`mps`) automatically. If it errors on `mps`, add `--device cpu`.
An M-series CPU is still several times faster than the cloud machine used here.

Then send back `models/buoy_zone_yolo11n.pt` and `results/zeroshot/` (or push them to a branch).

## How long

| Machine | 1 epoch (6,100 tiles) | 30 epochs |
|---|---|---|
| Cloud container, 4-core Xeon CPU (measured) | ~34 min | ~17 h (too slow) |
| Apple M1/M2/M3, Metal `mps` (estimate) | ~3–6 min | ~1.5–3 h; early stopping usually ends it sooner |
| Colab T4 / A100 (estimate) | ~1 min / <30 s | ~30 min / ~15 min (`colab/train_buoy_zone.ipynb`) |

To make it shorter, use `--epochs 15`, or `--fraction 0.5` (half of the tiles).

## Knobs

- More pasted buoys per tile: `python tools/make_zone_dataset.py --paste-per-tile 2`.
- Bigger model, if the small one misses far buoys: `--model yolo11s.pt` (about 3x slower).
