#!/usr/bin/env python3
"""
Build neg/ folders for each species by sampling from other species' pos/ files.

For each target species, neg count matches pos count (1:1 balance).
Negatives are sampled equally across all donor species; any shortfall from
small-pool donors is redistributed to the largest available pools.

Usage:
    cd /workspaces/non-avian-ml
    pixi run python src/build_negatives.py
    pixi run python src/build_negatives.py --datatype data_5s
    pixi run python src/build_negatives.py --dry-run
"""

import argparse
import random
import shutil
from pathlib import Path


DATA_ROOT = Path("/workspaces/non-avian-ml/data")
SEED = 42


def get_pos_files(data_root: Path, datatype: str) -> dict[str, list[Path]]:
    pos_files = {}
    for species_dir in sorted(data_root.iterdir()):
        if not species_dir.is_dir():
            continue
        pos_dir = species_dir / datatype / "pos"
        if pos_dir.exists():
            files = sorted(pos_dir.glob("*.wav"))
            if files:
                pos_files[species_dir.name] = files
    return pos_files


def sample_negatives(
    target: str,
    pos_files: dict[str, list[Path]],
    rng: random.Random,
) -> list[tuple[Path, str]]:
    """
    Sample n_needed files from donor species, equal quota with redistribution.
    Returns list of (source_path, donor_species_name).
    """
    n_needed = len(pos_files[target])
    donors = [s for s in pos_files if s != target]

    if not donors:
        raise ValueError(f"No donor species available for {target}")

    # Initial equal quota
    base = n_needed // len(donors)
    quota = {d: base for d in donors}
    remainder = n_needed - sum(quota.values())

    # Distribute remainder to largest pools first
    donors_by_size = sorted(donors, key=lambda d: len(pos_files[d]), reverse=True)
    for i in range(remainder):
        quota[donors_by_size[i % len(donors_by_size)]] += 1

    sampled: list[tuple[Path, str]] = []
    shortfall = 0

    for donor, n_quota in quota.items():
        pool = pos_files[donor]
        n_take = min(n_quota, len(pool))
        shortfall += n_quota - n_take
        sampled += [(f, donor) for f in rng.sample(pool, n_take)]

    # Fill shortfall from donors with remaining capacity, largest first
    if shortfall > 0:
        print(f"  NOTE: shortfall of {shortfall} for {target}, redistributing...")
        already_used = {donor: count for donor, count in quota.items()}
        for donor in donors_by_size:
            if shortfall <= 0:
                break
            used = already_used[donor]
            available = len(pos_files[donor]) - used
            if available <= 0:
                continue
            extra = min(available, shortfall)
            pool = [f for f in pos_files[donor] if (f, donor) not in sampled]
            sampled += [(f, donor) for f in rng.sample(pool, extra)]
            shortfall -= extra

    if shortfall > 0:
        print(f"  WARNING: could only get {len(sampled)}/{n_needed} negs for {target}")

    return sampled


def build_negatives(datatype: str = "data", dry_run: bool = False) -> None:
    pos_files = get_pos_files(DATA_ROOT, datatype)

    if not pos_files:
        print(f"No species found with {datatype}/pos/*.wav under {DATA_ROOT}")
        return

    print(f"Found {len(pos_files)} species with pos files ({datatype}):\n")
    for species, files in pos_files.items():
        print(f"  {species}: {len(files)} pos files")

    print()
    rng = random.Random(SEED)

    for target in pos_files:
        n_pos = len(pos_files[target])
        neg_dir = DATA_ROOT / target / datatype / "neg"

        if neg_dir.exists() and any(neg_dir.glob("*.wav")):
            print(f"{target}: neg/ already has files, skipping (delete manually to rebuild)")
            continue

        sampled = sample_negatives(target, pos_files, rng)

        print(f"{target}: {n_pos} pos → sampling {len(sampled)} negs")

        if not dry_run:
            neg_dir.mkdir(parents=True, exist_ok=True)
            for src, donor in sampled:
                shutil.copy2(src, neg_dir / src.name)
        else:
            donor_counts = {}
            for _, donor in sampled:
                donor_counts[donor] = donor_counts.get(donor, 0) + 1
            for donor, count in sorted(donor_counts.items()):
                print(f"    {donor}: {count}")

    if dry_run:
        print("\n[dry-run] No files were copied.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build neg/ folders from cross-species pos/ files.")
    parser.add_argument("--datatype", default="data", help="Subfolder name: 'data' or 'data_5s' (default: data)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without copying files")
    args = parser.parse_args()

    build_negatives(datatype=args.datatype, dry_run=args.dry_run)
