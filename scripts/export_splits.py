"""Writes data/splits/ (described in data/splits/README.md): the train/test split behind results/ and, for every
experiment, exactly which clips it trained and tested on, so the work can be reproduced, or the same clips reused,
without this code.

Each experiment still computes its clips from the manifests and config.yaml, through the same functions this script
calls. `--check` confirms they still match the files, and the pixi experiment tasks run it first.

Usage:
    pixi run splits           # write data/splits/
    pixi run check-splits     # exit 1 if data/splits/ differs from what the experiments would use now
"""
import argparse
import gzip
import sys

import pandas as pd

from core.binary import save_csv
from core.config import load_config
from core.paths import CONFIG, DATA
from core.study import load_study
from experiments import cross_detection, multiclass, window_length

OUT = DATA / "splits"


def partition_rows(clip_set, study) -> list[dict]:
    """Every clip: train, test, or unused (from a test recording but not sampled into its class's test set)."""
    p = study.partition
    test = set().union(*p.test_clips.values())
    return [{"clip_set": clip_set, "filename": f, "class": c, "source": p.source[f],
             "split": "test" if f in test else "unused" if p.source[f] in p.test_recordings else "train"}
            for c, f in zip(study.manifest.cls, study.manifest.filename)]


def clip_rows(study, experiment, set_, files, target=None, n=None, seed=None, labels=None) -> list[dict]:
    labels = [None] * len(files) if labels is None else labels
    return [{"experiment": experiment, "target": target, "n": n, "seed": seed, "set": set_, "filename": f,
             "class": study.cls_of[f], "label": y} for f, y in zip(files, labels)]


def binary_rows(experiment, study, sizes, seeds) -> list[dict]:
    """Saturation and window length: per target class, its test set and one draw per N and seed."""
    rows = []
    for c, s in study.splits.items():
        if not s.test_pos:
            continue
        rows += clip_rows(study, experiment, "test", s.test_files, c, labels=s.test_labels)
        for n in [n for n in sizes if n <= s.max_n]:
            for seed, (files, labels) in zip(seeds, s.draws(n, seeds)):
                rows += clip_rows(study, experiment, "train", files, c, n, seed, labels)
    return rows


def cross_detection_rows(cfg, study) -> tuple[list[dict], list[dict]]:
    """Both cross-detection versions: the shared test set and each detector's draws, plus the cutoff check's folds."""
    rows, folds = [], []
    for version, all_data in cross_detection.VERSIONS.items():
        experiment = "cross_detection" if version == "headline" else "cross_detection_all_data"
        _, test_files, _, sizes, draws = cross_detection.design(cfg, study, all_data)
        rows += clip_rows(study, experiment, "test", test_files)
        for c, d in draws.items():
            for seed, (files, labels) in zip(cfg.random_seeds, d):
                rows += clip_rows(study, experiment, "train", files, c, sizes[c], seed, labels)
                groups = [study.partition.source[f] for f in files]
                fold_of = {i: k for k, (_, va) in enumerate(cross_detection.cutoff_folds(labels, groups, seed))
                           for i in va}
                folds += [{"experiment": experiment, "target": c, "n": sizes[c], "seed": seed, "filename": f,
                           "fold": fold_of[i]} for i, f in enumerate(files) if i in fold_of]
    return rows, folds


def multiclass_rows(cfg, study) -> list[dict]:
    classes, _, test_files, _, draws = multiclass.design(cfg, study)
    rows = clip_rows(study, "multiclass", "test", test_files)
    for seed, (files, _) in zip(cfg.random_seeds, draws):
        rows += clip_rows(study, "multiclass", "train", files, n=cfg.headline_n, seed=seed)
    return rows


def tables(cfg) -> dict[str, pd.DataFrame]:
    main, windows = load_study(cfg), window_length.load_window_study(cfg)
    cross, folds = cross_detection_rows(cfg, main)
    experiments = (binary_rows("saturation", main, cfg.training_sizes, cfg.random_seeds) + cross
                   + multiclass_rows(cfg, main)
                   + binary_rows("window_length", windows, window_length.SIZES, cfg.random_seeds))
    ints = {"n": "Int64", "seed": "Int64", "label": "Int64"}  # empty where they don't apply
    return {"partition.csv": pd.DataFrame(partition_rows("main", main) + partition_rows("windows", windows)),
            "experiments.csv.gz": pd.DataFrame(experiments).astype(ints),
            "cutoff_folds.csv.gz": pd.DataFrame(folds)}


def read_text(path) -> str:
    return gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()


def main():
    parser = argparse.ArgumentParser(description="Export the train/test clips of every experiment to data/splits/.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--check", action="store_true", help="Compare with data/splits/ instead of writing.")
    args = parser.parse_args()
    out = tables(load_config(args.config))
    if args.check:
        stale = [f for f, df in out.items() if not (OUT / f).exists() or read_text(OUT / f) != df.to_csv(index=False)]
        if stale:
            print(f"{OUT} differs from the clips the experiments would use now: {', '.join(stale)}.\n"
                  "If you changed the data or the split settings on purpose, run `pixi run splits` and commit "
                  "data/splits/ with the new results.")
            sys.exit(1)
        print(f"OK: {OUT} matches the clips the experiments use.")
        return
    for f, df in out.items():
        path = save_csv(df, OUT / f)
        print(f"{path}: {len(df):,} rows, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
