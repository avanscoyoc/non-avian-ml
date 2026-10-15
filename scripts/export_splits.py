"""Writes the exact train/test splits behind results/ to data/splits/, so they can be reused without this code
(data/splits/README.md): every clip's place in the partition, and every test set and training draw (target x N x
seed) of the main clip set and the window-length clip set. The split itself is still computed from the manifests
and config.yaml by every experiment; --check confirms that it still matches the exported files.

Usage:
    pixi run python -m scripts.export_splits            # write data/splits/
    pixi run python -m scripts.export_splits --check    # exit 1 if data/splits/ differs from the current split
"""
import argparse
import gzip
import sys

import pandas as pd

from core.binary import save_csv
from core.config import load_config
from core.paths import CONFIG, DATA
from core.study import load_study, make_study
from experiments.window_length import CLASSES, MIN_TRAIN, SIZES, TEST_SIZE, WINDOW_DATA

OUT = DATA / "splits"


def window_study(cfg):
    """The window-length experiment's clip set and split, as experiments.window_length builds it."""
    m = pd.read_csv(WINDOW_DATA / "manifest.csv")
    manifest = pd.DataFrame({"cls": m.common_name, "filename": m.clip_id + ".wav", "source": m.source_audio_path})
    return make_study(manifest, CLASSES, cfg, TEST_SIZE, MIN_TRAIN)


def split_tables(name, study, sizes, seeds, all_data=False) -> tuple[list[dict], list[dict]]:
    """(partition rows, set rows) for one clip set. Training draws at every size the target's pools allow, plus
    its whole pool (N = max_n, the cross-detection --all-data draws) if `all_data`."""
    p = study.partition
    test = set().union(*p.test_clips.values())
    partition = [{"study": name, "filename": f, "class": c, "source": p.source[f],
                  "split": "test" if f in test else "unused" if p.source[f] in p.test_recordings else "train"}
                 for c, f in zip(study.manifest.cls, study.manifest.filename)]
    sets = []
    for c, s in study.splits.items():
        sets += [{"study": name, "target": c, "set": "test", "n": None, "seed": None, "filename": f, "label": y}
                 for f, y in zip(s.test_pos + s.test_neg, [1] * len(s.test_pos) + [0] * len(s.test_neg))]
        for n in sorted({n for n in sizes if n <= s.max_n} | ({s.max_n} if all_data and s.max_n else set())):
            for seed, (files, labels) in zip(seeds, s.draws(n, seeds)):
                sets += [{"study": name, "target": c, "set": "train", "n": n, "seed": seed, "filename": f, "label": y}
                         for f, y in zip(files, labels)]
    return partition, sets


def tables(cfg) -> dict[str, pd.DataFrame]:
    main = split_tables("main", load_study(cfg), cfg.training_sizes, cfg.random_seeds, all_data=True)
    windows = split_tables("windows", window_study(cfg), SIZES, cfg.random_seeds)
    sets = pd.DataFrame(main[1] + windows[1]).astype({"n": "Int64", "seed": "Int64"})  # test rows: empty n, seed
    return {"partition.csv": pd.DataFrame(main[0] + windows[0]), "sets.csv.gz": sets}


def read_text(path) -> str:
    return gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()


def main():
    parser = argparse.ArgumentParser(description="Export the train/test splits to data/splits/.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--check", action="store_true", help="Compare with data/splits/ instead of writing.")
    args = parser.parse_args()
    out = tables(load_config(args.config))
    if args.check:
        stale = [f for f, df in out.items() if not (OUT / f).exists() or read_text(OUT / f) != df.to_csv(index=False)]
        if stale:
            print(f"{OUT} differs from the split computed now: {', '.join(stale)}")
            sys.exit(1)
        print(f"OK: {OUT} matches the split computed from config.yaml and the manifests.")
        return
    for f, df in out.items():
        path = save_csv(df, OUT / f)
        print(f"{path}: {len(df):,} rows, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
