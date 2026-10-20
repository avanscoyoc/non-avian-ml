"""Writes results/listening_shortlist.csv: the clips most worth listening to in results/clip_review.csv, a few per
flagged recording, as a clip list for the jupyter_bioacoustic annotator (github.com/avanscoyoc/bioacoustic-review).

A clip is flagged when at least --min-models of the models place it nearest another class (`models_confused` in
clip_review.csv). Labels usually go wrong for a whole recording at once, so each recording with flagged clips gets
up to --per-recording of them (most confused first), and recordings with the most flagged clips come first.

Columns: id, path, start_time, end_time (what the annotator reads: index column `id`, audio column `path`): the
source recording as in data/metadata.csv (`source_audio_path`, under --audio-root if given) and the clip's 3 s window
in it (`start_time_3s`, `end_time_3s`), so the annotator plays the original recording with context around the clip.
Then class, split, filename, recording_clips and recording_flagged (the class's clips in that recording, and how
many are flagged), models_confused, confused_with, disagree (as in clip_review.csv).

Usage:
    pixi run python -m scripts.clip_review              # first, if results/clip_review.csv is missing
    pixi run python -m scripts.listening_shortlist
    pixi run python -m scripts.listening_shortlist --classes power_tools device_static --audio-root /Volumes/audio
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from core.binary import save_csv
from core.config import load_config
from core.paths import CONFIG

CLASSES = ["power_tools", "generator", "human_vocal"]
COLUMNS = ["id", "path", "start_time", "end_time", "class", "split", "filename", "recording_clips",
           "recording_flagged", "models_confused", "confused_with", "disagree"]


def shortlist(review: pd.DataFrame, classes: list[str], min_models: int, per_recording: int) -> pd.DataFrame:
    """Up to per_recording flagged clips per (class, recording), most confused first, most-flagged recordings first."""
    r = review[review["class"].isin(classes)].copy()
    group = r.groupby(["class", "source"])
    r["recording_clips"] = group["filename"].transform("size")
    r["recording_flagged"] = group["models_confused"].transform(lambda s: int((s >= min_models).sum()))
    r = r[r.models_confused >= min_models].sort_values(
        ["recording_flagged", "class", "source", "models_confused", "disagree"],
        ascending=[False, True, True, False, False])
    return r.groupby(["class", "source"], sort=False).head(per_recording).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Clips worth listening to, as a clip list for the annotator.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--classes", nargs="+", default=CLASSES, help=f"Default: {' '.join(CLASSES)}.")
    parser.add_argument("--min-models", type=int, default=4,
                        help="Flag a clip when at least this many models confuse it (default 4).")
    parser.add_argument("--per-recording", type=int, default=3, help="Flagged clips per recording (default 3).")
    parser.add_argument("--audio-root", help="Folder the metadata.csv source paths are relative to, as the annotator "
                                             "should see it (default: the paths as written in metadata.csv).")
    args = parser.parse_args()
    cfg = load_config(args.config)

    review = pd.read_csv(cfg.results_path / "clip_review.csv")
    unknown = sorted(set(args.classes) - set(review["class"]))
    if unknown:
        sys.exit(f"Not in clip_review.csv: {', '.join(unknown)}")
    out = shortlist(review, args.classes, args.min_models, args.per_recording)

    meta = pd.read_csv(cfg.clips_csv, usecols=["filename", "source_audio_path", "start_time_3s", "end_time_3s"])
    meta = meta.set_index("filename").loc[out.filename]
    paths = meta.source_audio_path
    out = out.assign(id=range(1, len(out) + 1),
                     path=[(Path(args.audio_root) / p).as_posix() for p in paths] if args.audio_root else paths.values,
                     start_time=meta.start_time_3s.round(3).values,
                     end_time=meta.end_time_3s.round(3).values)

    path = save_csv(out[COLUMNS], cfg.results_path / "listening_shortlist.csv")
    print(out.groupby("class", sort=False).agg(recordings=("source", "nunique"), clips=("filename", "size")))
    print(f"Written: {path} ({len(out)} clips, {(out.end_time - out.start_time).sum() / 60:.1f} min of audio)")


if __name__ == "__main__":
    main()
