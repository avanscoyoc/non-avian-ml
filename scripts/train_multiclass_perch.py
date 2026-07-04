"""
train_multiclass_perch.py

Trains multiclass Perch MLP classifiers for two ecologically coherent groups:
  anuran   : 4 anuran species
  nonbiotic: 6 abiotic/anthropogenic classes

Matches the class definitions and train/test counts used in
experiment/multiclass_fit.py (N_TRAIN=80, N_TEST=50, SPLIT_SEED=42).
Trains a single classifier (seed=1) and saves a deployable bundle.

Usage:
    python scripts/train_multiclass_perch.py
    python scripts/train_multiclass_perch.py --group anuran
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.model_loader import PerchModel
from src.results import save_classifier_bundle

# ---------------------------------------------------------------------------
# Class groups — must match experiment/multiclass_fit.py
# ---------------------------------------------------------------------------

GROUPS = {
    "anuran": [
        "pacific_chorus_frog",
        "woodhouses_toad",
        "yellow_legged_frog",
        "american_bullfrog",
    ],
    "nonbiotic": [
        "engine",
        "generator",
        "traffic",
        "device_static",
        "wind",
        "power_tools",
    ],
}

DATA_PATH    = Path("/workspaces/non-avian-ml/data")
MODEL_PATH   = Path("/workspaces/non-avian-ml/models/perch_8")
RESULTS_PATH = Path("/workspaces/non-avian-ml/results")
DATATYPE     = "data_5s"
N_TRAIN      = 80
N_TEST       = 50
SPLIT_SEED   = 42
TRAIN_SEED   = 1
BATCH_SIZE   = 32
N_EPOCHS     = 20

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_split(classes: list[str]) -> tuple[dict, dict]:
    """Deterministic recording-level train/test split (SPLIT_SEED)."""
    test_by_class, pool_by_class = {}, {}
    for cls in classes:
        pos = sorted((DATA_PATH / cls / DATATYPE / "pos").glob("*.wav"))
        if len(pos) < N_TEST + N_TRAIN:
            raise ValueError(f"{cls}: {len(pos)} clips, need >= {N_TEST + N_TRAIN}")
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(pos)
        test_by_class[cls] = [str(f) for f in pos[:N_TEST]]
        pool_by_class[cls] = [str(f) for f in pos[N_TEST:]]
    return test_by_class, pool_by_class


def extract_all(model: PerchModel, files: list[str]) -> torch.Tensor:
    embs = []
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(files)}")
        embs.append(model.extract_embeddings(fp).flatten())
    return torch.stack(embs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_classifier(dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(clf: nn.Module, embs: torch.Tensor, labels: list[int], device) -> nn.Module:
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    clf: nn.Module,
    embs: torch.Tensor,
    labels: list[int],
    label_names: list[str],
    device,
) -> dict:
    clf.eval()
    with torch.no_grad():
        logits = clf(embs.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    y = np.array(labels)
    acc = accuracy_score(y, preds)
    ovr_auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(y, preds)
    print(f"      accuracy={acc:.4f}  macro OVR AUC={ovr_auc:.4f}")
    print(classification_report(y, preds, target_names=label_names, digits=3))
    return {"accuracy": acc, "ovr_auc": ovr_auc, "cm": cm}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, label_names: list[str], output_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = cm / row_sums * 100
    display_names = [n.replace("_", " ").capitalize() for n in label_names]

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})"

    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.9), max(7, n * 0.75)))

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        sns.heatmap(
            cm_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
            xticklabels=display_names, yticklabels=display_names,
            linewidths=0.4, linecolor="#e0e0e0",
            cbar_kws={"label": "Recall (%)", "shrink": 0.6, "pad": 0.02},
            ax=ax, annot_kws={"size": 7.5, "linespacing": 1.3},
        )
        ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
        ax.set_ylabel("True", fontsize=10, labelpad=6)
        ax.tick_params(axis="x", labelsize=8.5, length=0)
        ax.tick_params(axis="y", labelsize=8.5, length=0, rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Recall (%)", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Per-group training
# ---------------------------------------------------------------------------

def run_group(group: str, classes: list[str], perch: PerchModel, device):
    print(f"\n{'='*60}")
    print(f"  group={group}  model=perch  seed={TRAIN_SEED}")

    test_by_class, pool_by_class = make_split(classes)

    test_files  = [f for cls in classes for f in test_by_class[cls]]
    test_labels = [i for i, cls in enumerate(classes) for _ in test_by_class[cls]]

    print(f"  Extracting test embeddings ({len(test_files)} clips)...")
    test_embs = extract_all(perch, test_files)

    rng = random.Random(TRAIN_SEED)
    train_parts, train_labels = [], []
    for i, cls in enumerate(classes):
        chosen = rng.sample(pool_by_class[cls], N_TRAIN)
        print(f"  Extracting train embeddings for {cls} ({N_TRAIN} clips)...")
        train_parts.append(extract_all(perch, chosen))
        train_labels.extend([i] * N_TRAIN)

    train_embs = torch.cat(train_parts)

    dim = test_embs.shape[1]
    clf = build_classifier(dim, len(classes)).to(device)
    print(f"\n  Training {len(classes)}-class classifier...")
    clf = train_classifier(clf, train_embs, train_labels, device)

    print("\n  Evaluating on test set...")
    metrics = evaluate_classifier(clf, test_embs, test_labels, classes, device)

    figs_path = RESULTS_PATH.parent / "figs"
    plot_confusion_matrix(
        metrics["cm"], classes,
        figs_path / f"confusion_matrix_multiclass_{group}_perch.png",
    )

    print(f"\n  Saving bundle: multiclass_{group}_perch ...")
    save_classifier_bundle(
        classifier=clf,
        embedding_model=perch,
        model_name="perch",
        species=classes,
        training_size=N_TRAIN,
        random_seed=TRAIN_SEED,
        test_auc=metrics["ovr_auc"],
        n_epochs=N_EPOCHS,
        results_path=str(RESULTS_PATH),
        bundle_name=f"multiclass_{group}_perch",
        labels={str(i): cls for i, cls in enumerate(classes)},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS), default=None,
                        help="Run a single group only (default: both)")
    args = parser.parse_args()

    groups_to_run = {args.group: GROUPS[args.group]} if args.group else GROUPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading Perch model...")
    perch = PerchModel(str(MODEL_PATH))

    for group, classes in groups_to_run.items():
        run_group(group, classes, perch, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
