"""Verifies data/clips/ against data/metadata.csv: every clip present at data/clips/{common_name}/{filename} and
no extra wav files; clips that are not exactly 3 s are listed. Prints per-class counts and Table 1 of
docs/brief.md (recordings and clips per class in the shared test/train split). experiments.saturation runs the
same check and stops if it fails.

Usage:
    pixi run check-data
"""
import argparse
import sys

from core.config import load_config
from core.data import check_data
from core.paths import CONFIG
from core.plotting import class_name
from core.study import load_study


def partition_table(study, cfg) -> str:
    """Markdown table: recordings and clips per class in the shared test/train split."""
    manifest, p = study.manifest, study.partition
    types = {c: t for t, classes in cfg.class_types.items() for c in classes}
    lines = ["| Class | Type | Recordings / clips | Test recordings / clips | Training recordings / clips | Max N |",
             "|---|---|---|---|---|---|"]
    for c in cfg.classes:
        d = manifest[manifest.cls == c]
        test_recs = {p.source[f] for f in p.test_clips[c]}
        train_recs = {p.source[f] for f in p.train_clips[c]}
        max_n = max([n for n in cfg.training_sizes if n <= len(p.train_clips[c])], default=0)
        name = class_name(c) + (" *" if c in p.low_sample else "")
        lines.append(f"| {name} | {types.get(c, '')} | {d.source.nunique()} / {len(d)} | "
                     f"{len(test_recs)} / {len(p.test_clips[c])} | {len(train_recs)} / {len(p.train_clips[c])} | {max_n} |")
    n_recs = manifest.source.nunique()
    lines.append(f"\n\\* low sample: about half of the class's recordings are held out for testing (see README).\n"
                 f"{len(p.test_recordings)} of {n_recs} recordings are test; the other "
                 f"{n_recs - len(p.test_recordings)} are training (all classes, including negative-only ones).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check data/clips/ against data/metadata.csv.")
    parser.add_argument("--config", default=CONFIG)
    cfg = load_config(parser.parse_args().config)
    study = load_study(cfg)
    manifest = study.manifest
    counts = manifest.groupby("cls").agg(clips=("filename", "size"), recordings=("source", "nunique"))
    print(counts.to_string())
    print(f"total: {len(manifest)} clips, {manifest.source.nunique()} source recordings\n")
    problems, notes = check_data(manifest, cfg.data_path)
    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if problems:
        sys.exit(1)
    print(f"OK: all {len(manifest)} clips present in {cfg.data_path}, no extra files.\n")
    print("Shared test/train split (config.yaml):\n")
    print(partition_table(study, cfg))


if __name__ == "__main__":
    main()
