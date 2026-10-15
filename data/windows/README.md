# data/windows

Clips for testing BirdNET+ V3.0 at 3 s, 5 s and 10 s window sizes (`experiments/window_length.py`), cut with ffmpeg from the source recordings at their native sample rate by `scripts/make_window_clips.py`. The source recordings (`audio/`) are not in the repository.

## Files

| File / folder | Contents |
| --- | --- |
| `../metadata.csv` | One row per annotated event (the main clip manifest) |
| `3s/`, `5s/`, `10s/` | Clips: `{common_name}/{clip_id}.wav`; the same `clip_id` appears in all three (not in git) |
| `manifest.csv` | Kept clips, with window bounds and file paths |
| `feasibility_all_clips.csv` | Every `metadata.csv` row with its `status` |

## How `metadata.csv` was used

- `source_audio_path` locates the recording in `audio/`.
- `start_time` and `end_time` give the event in the recording; all three windows are centered on its midpoint, `(start_time + end_time) / 2`.
- `common_name` sets the class folder and `filename` sets the clip name.

## What was kept or lost

Each row of `metadata.csv` gets one `status` in `feasibility_all_clips.csv`:

| Status | Meaning |
| --- | --- |
| `ok` | Kept: all three windows extracted |
| `too_close_to_edge` | Lost: the 10 s window runs past the start or end of the recording |
| `overlaps_10s` | Lost: its 10 s window overlaps an earlier kept clip from the same recording (named in `overlaps_with`) |
| `source_missing` | Lost: recording not found |

Checking at 10 s covers all sizes: since the windows are nested, a 10 s window that fits and doesn't overlap means the 3 s and 5 s windows don't either. Overlaps were dropped to avoid shared audio between clips.

## Kept clips by class

| Class | Clips |
| --- | ---: |
| pacific_chorus_frog | 705 |
| woodhouses_toad | 560 |
| field_cricket | 539 |
| american_bullfrog | 378 |
| device_static | 239 |
| engine | 135 |
| yellow_legged_frog | 125 |
| generator | 79 |
| wind | 79 |
| power_tools | 79 |
| coyote | 68 |
| human_vocal | 57 |
| traffic | 48 |
| metal_clanging | 12 |
| airplane | 11 |
| nutria | 7 |
| gun | 5 |
| fireworks | 3 |
| dog | 3 |
| water | 1 |
| thunder | 1 |
| human_non_vocal | 1 |
| **Total** | **3,135** |
