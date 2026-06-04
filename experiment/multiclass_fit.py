"""
multiclass_fit.py

Trains N=80-per-class multiclass MLP classifiers for two groups:
  anuran   : 4 classes (PCF, Woodhouse's toad, yellow-legged frog, bullfrog)
  nonbiotic: 5 classes (engine, generator, traffic, device_static, wind)

Models: birdnet, perch  |  Seeds: 1–10  →  40 fits total

Embeddings for each group×model are extracted ONCE (pre-seeded split), then
train clips are resampled per seed so extraction cost is not multiplied by seeds.

Outputs saved to results/experiment/:
  run_multiclass_{group}_{model}_seed{seed}.csv   per-class OVR AUC + F1
  cm_multiclass_{group}_{model}_seed{seed}.npy    confusion matrix (counts)
  labels_{group}.json                             class names (index order)

Usage:
    python experiment/multiclass_fit.py
    python experiment/multiclass_fit.py --group anuran --model birdnet
"""

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_loader import load_model

# ── Fixed design ──────────────────────────────────────────────────────────────
ANURAN    = ["pacific_chorus_frog", "woodhouses_toad", "yellow_legged_frog", "american_bullfrog"]
NONBIOTIC = ["engine", "generator", "traffic", "device_static", "wind", "power_tools"]
GROUPS    = {"anuran": ANURAN, "nonbiotic": NONBIOTIC}

MODEL_DATATYPE = {"birdnet": "data", "perch": "data_5s"}

N_TRAIN    = 80
N_TEST     = 50
SPLIT_SEED = 42          # fixes train/test assignment (same across models)
SEEDS      = list(range(1, 11))
N_EPOCHS   = 20
BATCH_SIZE = 32

ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data"
RESULTS_PATH = ROOT / "results" / "experiment"


# ── Data ─────────────────────────────────────────────────────────────────────

def make_split(classes: list[str], datatype: str) -> tuple[dict, dict]:
    """Deterministic recording-level train/test split (SPLIT_SEED).

    Returns test_by_class and pool_by_class dicts mapping class name → file list.
    """
    test_by_class, pool_by_class = {}, {}
    for cls in classes:
        pos = sorted((DATA_PATH / cls / datatype / "pos").glob("*.wav"))
        if len(pos) < N_TEST + N_TRAIN:
            raise ValueError(
                f"{cls}: {len(pos)} clips found, need ≥ {N_TEST + N_TRAIN}"
            )
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(pos)
        test_by_class[cls] = [str(f) for f in pos[:N_TEST]]
        pool_by_class[cls] = [str(f) for f in pos[N_TEST:]]
    return test_by_class, pool_by_class


def extract_all(model, files: list[str]) -> torch.Tensor:
    embs = []
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(files)}")
        embs.append(model.extract_embeddings(fp).flatten())
    return torch.stack(embs)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_classifier(dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


def train_classifier(
    clf: nn.Module,
    embs: torch.Tensor,
    labels: list[int],
    device: torch.device,
) -> nn.Module:
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights = weights / weights.sum() * len(counts)

    ds     = TensorDataset(embs.to(device), labels_t.to(device))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    clf.train()
    for ep in range(N_EPOCHS):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(clf(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 5 == 0 or ep == N_EPOCHS - 1:
            print(f"      epoch {ep+1:>3}  loss={total/len(loader):.4f}")
    return clf


def evaluate_classifier(
    clf: nn.Module,
    embs: torch.Tensor,
    labels: list[int],
    label_names: list[str],
    device: torch.device,
) -> dict:
    clf.eval()
    with torch.no_grad():
        logits = clf(embs.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    y = np.array(labels)
    per_class_auc = roc_auc_score(y, probs, multi_class="ovr", average=None)
    cm     = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=label_names, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"      macro F1={macro_f1:.4f}  per-class AUC={per_class_auc.round(4)}")
    return {"per_class_auc": per_class_auc, "cm": cm, "report": report}


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_run(group: str, model_name: str, seed: int, label_names: list[str], metrics: dict):
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    report   = metrics["report"]
    macro_f1 = report["macro avg"]["f1-score"]
    rows = [
        {
            "group": group,
            "model": model_name,
            "seed": seed,
            "class_name": cls,
            "class_idx": i,
            "ovr_auc": metrics["per_class_auc"][i],
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
            "macro_f1":  macro_f1,
        }
        for i, cls in enumerate(label_names)
    ]
    csv_path = RESULTS_PATH / f"run_multiclass_{group}_{model_name}_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    np.save(RESULTS_PATH / f"cm_multiclass_{group}_{model_name}_seed{seed}.npy", metrics["cm"])

    labels_path = RESULTS_PATH / f"labels_{group}.json"
    if not labels_path.exists():
        labels_path.write_text(json.dumps(label_names))


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_group_model(group: str, classes: list[str], model_name: str):
    print(f"\n{'='*60}")
    print(f"  group={group}  model={model_name}")

    datatype = MODEL_DATATYPE[model_name]
    test_by_class, pool_by_class = make_split(classes, datatype)

    test_files  = [f for cls in classes for f in test_by_class[cls]]
    test_labels = [i for i, cls in enumerate(classes) for _ in test_by_class[cls]]

    print(f"  Loading {model_name}...")
    emb_model, _ = load_model(model_name)

    print(f"  Extracting test embeddings ({len(test_files)} clips)...")
    test_embs = extract_all(emb_model, test_files)

    pool_embs: dict[str, torch.Tensor] = {}
    for cls in classes:
        print(f"  Extracting pool for {cls} ({len(pool_by_class[cls])} clips)...")
        pool_embs[cls] = extract_all(emb_model, pool_by_class[cls])

    emb_model.cleanup()

    dim    = test_embs.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        print(f"\n  Seed {seed}/{len(SEEDS)}")
        rng = random.Random(seed)
        train_parts, train_labels = [], []
        for i, cls in enumerate(classes):
            pool = pool_embs[cls]
            idx  = torch.tensor(rng.sample(range(len(pool)), N_TRAIN))
            train_parts.append(pool[idx])
            train_labels.extend([i] * N_TRAIN)

        train_embs = torch.cat(train_parts)
        clf = build_classifier(dim, len(classes)).to(device)
        clf = train_classifier(clf, train_embs, train_labels, device)
        metrics = evaluate_classifier(clf, test_embs, test_labels, classes, device)
        save_run(group, model_name, seed, classes, metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS), default=None)
    parser.add_argument("--model", choices=list(MODEL_DATATYPE), default=None)
    args = parser.parse_args()

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS
    models_to_run = [args.model] if args.model else list(MODEL_DATATYPE)

    total = sum(len(SEEDS) for _ in groups_to_run for _ in models_to_run)
    print(f"Planned fits: {total}  (groups={list(groups_to_run)}  models={models_to_run}  seeds={SEEDS})")

    for group, classes in groups_to_run.items():
        for model_name in models_to_run:
            run_group_model(group, classes, model_name)

    print("\nDone. Aggregating results...")
    dfs = [pd.read_csv(p) for p in RESULTS_PATH.glob("run_multiclass_*.csv")]
    if dfs:
        agg = (
            pd.concat(dfs)
            .groupby(["group", "model", "class_name"])[["ovr_auc", "f1", "macro_f1"]]
            .agg(["mean", "std"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(RESULTS_PATH / "multiclass_results.csv", index=False)
        print(f"Aggregated results → {RESULTS_PATH / 'multiclass_results.csv'}")


if __name__ == "__main__":
    main()
