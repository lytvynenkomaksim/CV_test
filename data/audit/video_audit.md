# Video audit

Sharpness = mean var(Laplacian) on grayscale frames at 640px width; shake_px = median phase-correlation shift between consecutive sampled frames (px at 640px width, includes intentional panning); skier_ratio = share of sampled frames with a YOLO person (conf>thr). Quality tier: low if short side <560 or sharpness <150; high if short side >=900 and sharpness >=400; else medium.

| video | res | fps | dur_s | frames | kbps | codec | MB | sharpness | shake_px | quality | skier_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| data/videos/drive_extra/GreecePro_MenRd3__Aaron_Davis_Rd3_6at12.mp4 | 854x480 | 30.0 | 19.93 | 598 | 1711 | h264 | 4.3 | 238.9 | 35.16 | low | 0.975 |
| data/videos/drive_extra/GreecePro_WomenRd3__Allie_Nicholson_Rd3_6at12m.mp4 | 1280x720 | 60.0 | 20.61 | 1236 | 3155 | h264 | 8.1 | 372.8 | 43.91 | medium | 0.762 |
| data/videos/drive_extra/ItalyPro_OpenWomenRd2__Julie_Mishler_6at32_Rd2.mp4 | 1920x1064 | 29.57 | 20.5 | 605 | 8065 | h264 | 20.7 | 798.7 | 2.03 | high | 0.829 |
| data/videos/drive_extra/ItalyPro_SanGervasioD2_ProWomen__Alexandra_Garcia_6_at_38_off.mp4 | 1072x600 | 30.01 | 28.89 | 865 | 3718 | h264 | 13.4 | 214.9 | 20.43 | medium | 0.879 |
| data/videos/front_view/Damir Filaretov 1 at 39.5 off (close call).mp4 | 1070x596 | 30.04 | 15.26 | 456 | 4794 | h264 | 9.1 | 232.2 | 22.82 | medium | 0.839 |
| data/videos/front_view/Damir Filaretov 6 at 32 off.mp4 | 1074x600 | 30.01 | 25.5 | 764 | 3990 | h264 | 12.7 | 251.1 | 20.04 | medium | 0.882 |
| data/videos/front_view/Damir Filaretov 6 at 35 off.mp4 | 1074x600 | 29.98 | 24.68 | 739 | 4211 | h264 | 13.0 | 192.6 | 55.68 | medium | 0.74 |
| data/videos/front_view/Damir Filaretov 6 at 38 off.mp4 | 1074x598 | 30.01 | 26.24 | 787 | 4049 | h264 | 13.3 | 253.3 | 21.42 | medium | 0.906 |
| data/videos/front_view/Lucas Cornale 1 at 38 off (good crash).mp4 | 1074x600 | 30.01 | 18.37 | 549 | 4440 | h264 | 10.2 | 257.0 | 17.63 | medium | 0.595 |
| data/videos/front_view/Lucas Cornale 6 at 35 off.mp4 | 1066x602 | 29.97 | 25.45 | 760 | 4003 | h264 | 12.7 | 197.6 | 28.68 | medium | 0.824 |
| data/videos/front_view/Vincenzo Marino 2 at 39.5 off.mp4 | 1072x602 | 30.0 | 19.97 | 597 | 5315 | h264 | 13.3 | 189.6 | 21.18 | medium | 0.75 |
| data/videos/front_view/Vincenzo Marino 6 at 28 off.mp4 | 1072x598 | 29.99 | 25.1 | 751 | 4106 | h264 | 12.9 | 217.1 | 33.71 | medium | 0.804 |
| data/videos/front_view/Vincenzo Marino 6 at 32 off.mp4 | 1070x602 | 29.99 | 26.19 | 783 | 3932 | h264 | 12.9 | 246.8 | 16.65 | medium | 0.887 |
| data/videos/front_view/Vincenzo Marino 6 at 35 off.mp4 | 1066x598 | 30.01 | 24.64 | 736 | 4272 | h264 | 13.2 | 190.1 | 50.03 | medium | 0.776 |
