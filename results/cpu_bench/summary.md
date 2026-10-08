# Demo run: buoy=yolo-buoy, pose=yolo-s, skier=yolo11s, buoy detector every 1 frame(s)

| video | official | estimate | confirmed | seen | slots | skier_visible | pose_good | buoy_frames | buoy_hidden | hidden_gap_s | ms_per_frame | x_realtime | output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| data/videos/drive_extra/ItalyPro_SanGervasioD2_ProWomen__Alexandra_Garcia_6_at_38_off.mp4 | ? | 6 | 1 | 4 | L+ R~ L+ R+ L+ R~ | 0.901 | 0.877 | 0.621 | 0.283 | 0.6 | 497 | 14.91 | ItalyPro_SanGervasioD2_ProWomen__Alexandra_Garcia_6_at_38_off.mp4 |
| data/videos/drive_extra/GreecePro_MenRd3__Aaron_Davis_Rd3_6at12.mp4 | 6 | 5 | 5 | 5 | R+ L+ R+ L+ R+ L- | 0.938 | 0.89 | 0.117 | 0.068 | 0.07 | 509 | 15.27 | GreecePro_MenRd3__Aaron_Davis_Rd3_6at12.mp4 |
| data/videos/drive_extra/ItalyPro_OpenWomenRd2__Julie_Mishler_6at32_Rd2.mp4 | 6 | 6 | 1 | 3 | R+ L~ R+ L+ R? L~ | 0.889 | 0.866 | 0.628 | 0.181 | 0.47 | 529 | 15.65 | ItalyPro_OpenWomenRd2__Julie_Mishler_6at32_Rd2.mp4 |
| data/videos/youtube/youtube_regina_jaquess.mp4 | ? | 6 | 1 | 1 | L+ R? L? R? L? R? | 0.852 | 0.836 | 0.112 | 0.314 | 0.43 | 515 | 15.45 | youtube_regina_jaquess.mp4 |
