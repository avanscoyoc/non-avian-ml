"""
check_data.py

Verifies data/ against data/metadata.csv: every clip present at data/{common_name}/{filename}
and no extra wav files; clips that are not exactly 3 s are listed. Prints per-class counts
and the recording/clip breakdown of the shared test/train partition (as in the README).
main.py runs the same check and stops if it fails.

Usage:
    pixi run python scripts/check_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CLIPS_CSV, DATA, ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import check_data, load_manifest  # noqa: E402


def partition_table(manifest, cfg) -> str:
    """Markdown table: recordings and clips per class in the shared test/train partition."""
    p = build_partition(manifest, cfg)
    lines = ["| Class | Recordings / clips | Test recordings / clips | Training recordings / clips | Max N |",
             "|---|---|---|---|---|"]
    for c in cfg.species:
        d = manifest[manifest.cls == c]
        test_recs = {p.source[f] for f in p.test_clips[c]}
        train_recs = {p.source[f] for f in p.train_clips[c]}
        max_n = max([n for n in cfg.training_sizes if n <= len(p.train_clips[c])], default=0)
        lines.append(f"| {c.replace('_', ' ').capitalize()} | {d.source.nunique()} / {len(d)} | "
                     f"{len(test_recs)} / {len(p.test_clips[c])} | {len(train_recs)} / {len(p.train_clips[c])} | {max_n} |")
    n_recs = manifest.source.nunique()
    lines.append(f"\n{len(p.test_recordings)} of {n_recs} recordings are test; the other "
                 f"{n_recs - len(p.test_recordings)} are training.")
    return "\n".join(lines)


def main():
    manifest = load_manifest(CLIPS_CSV)
    counts = manifest.groupby("cls").agg(clips=("filename", "size"), recordings=("source", "nunique"))
    print(counts.to_string())
    print(f"total: {len(manifest)} clips, {manifest.source.nunique()} source recordings\n")
    problems, notes = check_data(manifest, DATA)
    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if problems:
        sys.exit(1)
    print(f"OK: all {len(manifest)} clips present in {DATA}, no extra files.\n")
    print("Shared test/train partition (src/config.yaml):\n")
    print(partition_table(manifest, load_config(ROOT / "src" / "config.yaml")))


if __name__ == "__main__":
    main()
