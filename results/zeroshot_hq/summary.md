Buoy detection inside the zone near the skier only. A hit = predicted centre within max(6 px, 0.75 x size) of a true buoy. P/R/F1 at the best-F1 threshold (thr). R_small = buoys under 12 px tall. ms = per zone crop (1280 px wide) on 4 CPU cores.

### All test frames: 28 hand-checked buoys, 40 frames, 5 videos

| model | frames | AP | P | R | F1 | thr | fp_per_frame | R_LEFT | R_MIDDLE | R_RIGHT | R_small | ms_per_crop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ensemble_vote | 40 | 0.74 | 0.68 | 0.82 | 0.74 | 0.25 | 0.28 | 0.81 (16) | 0.78 (9) | 1.00 (3) | 0.00 (2) | 31246.9 |
| yoloe26l | 40 | 0.56 | 0.82 | 0.64 | 0.72 | 0.08 | 0.1 | 0.56 (16) | 0.67 (9) | 1.00 (3) | 0.00 (2) | 2431.4 |
| colorblob_clip | 40 | 0.66 | 0.89 | 0.57 | 0.7 | 0.42 | 0.05 | 0.50 (16) | 0.56 (9) | 1.00 (3) | 0.50 (2) | 2311.3 |
| finetuned | 40 | 0.61 | 0.63 | 0.79 | 0.7 | 0.16 | 0.32 | 0.75 (16) | 0.78 (9) | 1.00 (3) | 0.00 (2) | 673.6 |
| ensemble_untrained | 40 | 0.66 | 0.57 | 0.82 | 0.68 | 0.25 | 0.42 | 0.81 (16) | 0.78 (9) | 1.00 (3) | 0.50 (2) | 30573.3 |
| ensemble_clip_only | 40 | 0.6 | 0.64 | 0.64 | 0.64 | 0.28 | 0.25 | 0.62 (16) | 0.56 (9) | 1.00 (3) | 0.00 (2) | 31246.9 |
| colorblob | 40 | 0.35 | 0.69 | 0.32 | 0.44 | 0.25 | 0.1 | 0.25 (16) | 0.44 (9) | 0.33 (3) | 1.00 (2) | 652.7 |

