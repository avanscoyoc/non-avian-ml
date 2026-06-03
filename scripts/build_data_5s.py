#!/usr/bin/env python3
"""
Create data_5s/ directories by padding each wav in data/ with 1s silence
before and 1s silence after (3s clip → 5s clip), preserving original sample rate.

Mirrors the full data/ structure (pos/ and neg/ subfolders) into data_5s/.

Usage:
    cd /workspaces/non-avian-ml
    pixi run python scripts/build_data_5s.py
    pixi run python scripts/build_data_5s.py --dry-run
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


DATA_ROOT = Path("/workspaces/non-avian-ml/data")
SRC_DATATYPE = "data"
DST_DATATYPE = "data_5s"


def pad_and_save(src: Path, dst: Path) -> None:
    audio, sr = sf.read(str(src), dtype="float32")
    pad_samples = sr  # 1 second of silence
    if audio.ndim == 1:
        silence = np.zeros(pad_samples, dtype=np.float32)
        padded = np.concatenate([silence, audio, silence])
    else:
        silence = np.zeros((pad_samples, audio.shape[1]), dtype=np.float32)
        padded = np.concatenate([silence, audio, silence], axis=0)
    sf.write(str(dst), padded, sr)


def build_data_5s(dry_run: bool = False) -> None:
    species_dirs = sorted(
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and (d / SRC_DATATYPE).exists()
    )

    if not species_dirs:
        print(f"No species found with '{SRC_DATATYPE}/' under {DATA_ROOT}")
        return

    for species_dir in species_dirs:
        src_base = species_dir / SRC_DATATYPE
        dst_base = species_dir / DST_DATATYPE

        subfolders = [p for p in src_base.iterdir() if p.is_dir()]
        if not subfolders:
            continue

        for subfolder in sorted(subfolders):  # pos/, neg/
            src_dir = subfolder
            dst_dir = dst_base / subfolder.name
            wav_files = sorted(src_dir.glob("*.wav"))

            if not wav_files:
                continue

            print(f"{species_dir.name}/{SRC_DATATYPE}/{subfolder.name} → {DST_DATATYPE}/{subfolder.name}: {len(wav_files)} files")

            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                for src_file in wav_files:
                    dst_file = dst_dir / src_file.name
                    if dst_file.exists():
                        continue
                    pad_and_save(src_file, dst_file)

    if dry_run:
        print("\n[dry-run] No files were written.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pad data/ wav files with 1s silence each side → data_5s/"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    args = parser.parse_args()

    build_data_5s(dry_run=args.dry_run)
