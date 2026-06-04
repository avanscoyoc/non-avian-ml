"""
extract_embeddings.py

Extracts test-set embeddings (50 clips per class) for both groups and both models.
Uses the same SPLIT_SEED and N_TEST as multiclass_fit.py for consistency.

Output (one file per group × model):
    results/experiment/embeddings_{group}_{model}.npz
      embeddings : float32 array (n_clips, embedding_dim)
      labels     : int array     (n_clips,)
      class_names: str array     (n_classes,)

Usage:
    python experiment/extract_embeddings.py [--group anuran|nonbiotic] [--model birdnet|perch]
"""

import argparse, random, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_loader import load_model

# ── Must match multiclass_fit.py ─────────────────────────────────────────────
ANURAN    = ["pacific_chorus_frog", "woodhouses_toad", "yellow_legged_frog", "american_bullfrog"]
NONBIOTIC = ["engine", "generator", "traffic", "device_static", "wind", "power_tools"]
GROUPS    = {"anuran": ANURAN, "nonbiotic": NONBIOTIC}

MODEL_DATATYPE = {"birdnet": "data", "perch": "data_5s"}

N_TEST     = 50
SPLIT_SEED = 42

ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data"
RESULTS_PATH = ROOT / "results" / "experiment"


def get_test_files(classes: list[str], datatype: str) -> tuple[list[str], list[int]]:
    """Return (files, labels) for the fixed test split."""
    files, labels = [], []
    for i, cls in enumerate(classes):
        pos = sorted((DATA_PATH / cls / datatype / "pos").glob("*.wav"))
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(pos)
        files.extend(str(f) for f in pos[:N_TEST])
        labels.extend([i] * N_TEST)
    return files, labels


def extract(model, files: list[str]) -> np.ndarray:
    embs = []
    for j, fp in enumerate(files):
        if (j + 1) % 100 == 0:
            print(f"    {j+1}/{len(files)}")
        embs.append(model.extract_embeddings(fp).flatten().numpy())
    return np.stack(embs).astype(np.float32)


def run_group_model(group: str, classes: list[str], model_name: str):
    print(f"\n  group={group}  model={model_name}")
    datatype = MODEL_DATATYPE[model_name]
    files, labels = get_test_files(classes, datatype)

    print(f"  Loading {model_name}, extracting {len(files)} clips...")
    emb_model, _ = load_model(model_name)
    embeddings = extract(emb_model, files)
    emb_model.cleanup()

    out = RESULTS_PATH / f"embeddings_{group}_{model_name}.npz"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        embeddings=embeddings,
        labels=np.array(labels, dtype=np.int32),
        class_names=np.array(classes),
    )
    print(f"  Saved {embeddings.shape} → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--model", choices=list(MODEL_DATATYPE), default=None)
    args = parser.parse_args()

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS
    models_to_run = [args.model] if args.model else list(MODEL_DATATYPE)

    for group, classes in groups_to_run.items():
        for model_name in models_to_run:
            run_group_model(group, classes, model_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
