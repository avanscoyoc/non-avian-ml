"""Writes data/metadata_noise_model.csv: the clips of data/metadata.csv that count toward the noise model's targets,
with the site and recording each comes from.

Keeps the noise classes only, and per class at most 1 clip per event (the 3 s window nearest the event's centre),
5 per recording and 10 per site (taken round-robin over the site's recordings, so a site's clips come from as many
recordings as possible). Where a cap drops clips, the choice is seeded random, so reruns give the same file.

Site and recording come from source_audio_path, by source folder:
  ExtremeSSD/Mojave/Birds/BD-31209A/<file>     site BD-31209A (A and B are different sites), recording BD-31209A/<file>
  samples_{devicestatic,fireworks,gun}/<file>  site = the code before the first "_" (COWO4, EDSA4b), recording =
                                               CODE_YEAR_DATE_TIME: the original recording the snippet was cut from
  samples_generator{1,2}/<file>                site = the location (Pumphouse, Windmill, ...), recording = the file

Columns: the 8 columns of data/metadata.csv, then site and recording_name.

Usage:
    pixi run python -m scripts.noise_model_metadata
"""
import argparse
import re

import numpy as np
import pandas as pd

from core.binary import save_csv
from core.paths import DATA

CLASSES = ["engine", "fireworks", "gun", "power_tools", "siren", "human_vocal", "human_non_vocal",
           "generator", "traffic", "airplane", "device_static", "wind", "water"]
MAX_PER_RECORDING = 5
MAX_PER_SITE = 10

MOJAVE = re.compile(r"ExtremeSSD/Mojave/Birds/(BD-\d+[AB])/([^/]+)\.\w+")
SAMPLES = re.compile(r"audio/samples_(?:devicestatic|fireworks_45per_bin|gun_15per_bin)/"
                     r"(([A-Za-z]+\d*[a-z]?)_\d{4}_\d{8}_\d{6})(?:_[^/]*)?\.\w+")
GENERATOR = re.compile(r"audio/samples_generator\d/(\d{4}_\d{2}_\d{2}_([A-Za-z0-9]+)_R\d+_[^/]+)\.\w+")


def site_and_recording(path: str) -> tuple[str, str]:
    if m := MOJAVE.fullmatch(path):
        return m[1], f"{m[1]}/{m[2]}"
    if m := SAMPLES.fullmatch(path):
        return m[2], m[1]
    if m := GENERATOR.fullmatch(path):
        return m[2], m[1]
    raise ValueError(f"no site rule for {path}")


def select(clips: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per class: 1 clip per event, then <= MAX_PER_RECORDING per recording and <= MAX_PER_SITE per site."""
    clips = clips.assign(r=np.random.default_rng(seed).random(len(clips)),
                         off=(clips.start_time_3s + 1.5 - (clips.start_time + clips.end_time) / 2).abs())
    event = ["common_name", "source_audio_path", "start_time", "end_time"]
    clips = clips.sort_values(["off", "r"]).groupby(event).head(1)
    clips["k"] = clips.sort_values("r").groupby(["common_name", "recording_name"]).cumcount()
    clips = clips[clips.k < MAX_PER_RECORDING]
    clips = clips.sort_values(["k", "r"]).groupby(["common_name", "site"]).head(MAX_PER_SITE)
    return clips.sort_index().drop(columns=["r", "off", "k"])


def main():
    parser = argparse.ArgumentParser(description="Clips of the noise classes that fit the caps, with site and recording.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_csv(DATA / "metadata.csv", float_precision="round_trip")  # copy the times exactly
    clips = meta[meta.common_name.isin(CLASSES)].copy()
    clips[["site", "recording_name"]] = clips.source_audio_path.map(site_and_recording).tolist()
    out = select(clips, args.seed)

    path = save_csv(out, DATA / "metadata_noise_model.csv")
    summary = out.groupby("common_name").agg(clips=("filename", "size"), sites=("site", "nunique"),
                                             recordings=("recording_name", "nunique"))
    print(summary.reindex(CLASSES).fillna(0).astype(int).to_string())
    print(f"Written: {path} ({len(out)} of {len(clips)} clips)")


if __name__ == "__main__":
    main()
