# Video sources

## front_view/
10 clips from the team Google Drive, folder `Italy Pro/San Gervasio Day 2/Men/...`
(Junior Boys etc.). Fixed wide camera on the towing boat looking back at the skier.

## drive_extra/
Extra test clips from the same Drive (download only, unchanged copies, no re-encode).
Filename = sanitized Drive folder + `__` + original file name.
Chosen to span quality / fps and to add women for a men-vs-women comparison.
See `data/audit/drive_folder_survey.md` for how each folder was classified.

| file | Drive path | why |
|---|---|---|
| ItalyPro_OpenWomenRd2__Julie_Mishler_6at32_Rd2.mp4 | Italy Pro/Open Women Round 2/Julie Mishler 6@32 Rd 2.mp4 | women; best quality found (1920x1064, ~8 Mbps), wide boat view with wakes |
| ItalyPro_SanGervasioD2_ProWomen__Alexandra_Garcia_6_at_38_off.mp4 | Italy Pro/San Gervasio Day 2/Women/Pro Women/Alexandra Garcia 6 at 38 off.mp4 | women; identical camera setup to front_view/ (direct men vs women comparison) |
| GreecePro_WomenRd3__Allie_Nicholson_Rd3_6at12m.mp4 | Greece Pro/Women Rd3/Allie Nicholson-Rd3-6@12m.mp4 | women; 1280x720 **60 fps**, zoomed boat camera |
| GreecePro_MenRd3__Aaron_Davis_Rd3_6at12.mp4 | Greece Pro/Men Rd3/Aaron Davis-Rd3-6@12.mp4 | men; lowest quality found (854x480, 1.7 Mbps), zoomed boat camera |

## youtube/
Not downloaded (2026-10-07): from this sandbox YouTube answers "Sign in to confirm
you're not a bot" for all player clients; with `player_client=web_embedded` + node JS
runtime the format list is returned, but every media request to googlevideo.com
returns HTTP 403 (no PO token / IP-bound URLs). Needs cookies or a PO-token provider.

Candidates checked by metadata + thumbnail (boat-mounted camera looking back at skier),
to fetch from a normal machine:

| URL | channel | title | max fmt |
|---|---|---|---|
| https://www.youtube.com/watch?v=riNiHLDZzlU | John Horton | Regina Jaquess pending Women's slalom world record of 4 buoys at 41 - Raw Video & boat path | 1080p30, 131 s (pylon in frame, raw boat cam) - best pick |
| https://www.youtube.com/watch?v=CHJiJiLx3Bc | Ski-Doc | Ski-Doc Camera Mount for Ski Nautique - test video | 1080p30, 38 s (wide pylon-mount view, both wakes) |
| https://www.youtube.com/watch?v=92qKEvK3nDw | wtrskrs | Nate Smith 41 off 20200614 | 1080p30, 26 s (zoomed boat cam) |
| https://www.youtube.com/watch?v=e5KsOJFiD7M | Learn WaterSports | Nate Smith - Men's Pro Slalom Finals - Masters 2025 | 1080p60, 475 s (broadcast, view not verified) |

Suggested command (with cookies):
`yt-dlp --cookies cookies.txt -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" URL`
