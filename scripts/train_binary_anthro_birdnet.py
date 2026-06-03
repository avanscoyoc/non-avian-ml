"""
train_binary_anthro_birdnet.py

Binary BirdNET classifier: anthropogenic (pos=1) vs natural (neg=0).
- Uses ALL available .wav files from each species' pos/ directory.
- Straight 80/20 train/test split, no k-fold, no per-class subsampling.
- CrossEntropyLoss with inverse-frequency class weights.

Usage:
    python scripts/train_binary_anthro_birdnet.py
"""

import math
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.model_loader import BirdNETModel
from src.results import save_classifier_bundle

# ---------------------------------------------------------------------------
# Species groups
# ---------------------------------------------------------------------------

ANTHROPOGENIC = [
    "device_static",
    "generator",
    "traffic",
    "engine",
    "power_tools",
    "human_vocal",
    "metal_clanging",
    "airplane",
    "gun",
    "fireworks",
    "human_non_vocal",
]

NATURAL = [
    "wind",
    "pacific_chorus_frog",
    "woodhouses_toad",
    "field_cricket",
    "yellow_legged_frog",
    "american_bullfrog",
    "nutria",
    "coyote",
    "dog",
    "water",
    "thunder",
]

DATA_PATH   = Path("/workspaces/non-avian-ml/data")
MODEL_PATH  = Path("/workspaces/non-avian-ml/models/birdnet_2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
RESULTS_PATH = Path("/workspaces/non-avian-ml/results")
DATATYPE    = "data"
SEED        = 1
BATCH_SIZE  = 32
N_EPOCHS    = 20

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def gather_files(species_list: list[str], label: int, datatype: str) -> tuple[list[str], list[int]]:
    files = []
    for sp in species_list:
        pos_dir = DATA_PATH / sp / datatype / "pos"
        wavs = sorted(pos_dir.glob("*.wav"))
        if not wavs:
            print(f"  WARNING: no .wav files found in {pos_dir}")
        files.extend(str(f) for f in wavs)
    return files, [label] * len(files)


def train_test_split_80_20(files: list[str], labels: list[int], seed: int):
    combined = list(zip(files, labels))
    rng = random.Random(seed)
    rng.shuffle(combined)
    n_test = math.floor(len(combined) * 0.2)
    test  = combined[:n_test]
    train = combined[n_test:]
    t_files, t_labels = zip(*train)
    v_files, v_labels = zip(*test)
    return list(t_files), list(t_labels), list(v_files), list(v_labels)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model: BirdNETModel, files: list[str]) -> torch.Tensor:
    embeddings = []
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(files)}")
        emb = model.extract_embeddings(fp)
        embeddings.append(emb.flatten())
    return torch.stack(embeddings)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_classifier(embedding_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(classifier, embeddings, labels, device):
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts   = torch.bincount(labels_t).float()
    weights  = (1.0 / counts.clamp(min=1)).to(device)
    weights  = weights / weights.sum() * 2  # normalise to mean=1

    dataset  = TensorDataset(embeddings.to(device), labels_t.to(device))
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    classifier.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for feats, lbls in loader:
            optimizer.zero_grad()
            loss = criterion(classifier(feats), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(f"    Epoch {epoch + 1}/{N_EPOCHS}  loss={epoch_loss / len(loader):.4f}")

    return classifier


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(classifier, embeddings, labels, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = np.argmax(logits.cpu().numpy(), axis=1)

    true = np.array(labels)
    auc  = roc_auc_score(true, probs)
    cm   = confusion_matrix(true, preds)
    print(f"\n  AUC: {auc:.4f}")
    print(classification_report(true, preds, target_names=["natural", "anthropogenic"], digits=3))
    return auc, cm


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    class_names = ["Natural", "Anthropogenic"]

    # Row-normalize to recall percentages; keep raw counts for annotations
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_pct = cm / row_sums * 100

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})"

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.4,
            linecolor="#e0e0e0",
            cbar_kws={"label": "Recall (%)", "shrink": 0.85, "pad": 0.02},
            ax=ax,
            annot_kws={"size": 10, "linespacing": 1.4},
        )

        ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
        ax.set_ylabel("True", fontsize=10, labelpad=6)
        ax.tick_params(axis="x", labelsize=9.5, length=0, rotation=0)
        ax.tick_params(axis="y", labelsize=9.5, length=0, rotation=0)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8.5)
        cbar.set_label("Recall (%)", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    pos_files, pos_labels = gather_files(ANTHROPOGENIC, label=1, datatype=DATATYPE)
    neg_files, neg_labels = gather_files(NATURAL,       label=0, datatype=DATATYPE)

    all_files  = pos_files + neg_files
    all_labels = pos_labels + neg_labels

    print(f"Total files: {len(all_files)}  (pos={len(pos_files)}, neg={len(neg_files)})")

    train_files, train_labels, test_files, test_labels = train_test_split_80_20(
        all_files, all_labels, seed=SEED
    )
    print(f"Train: {len(train_files)}  |  Test: {len(test_files)}\n")

    print("Loading BirdNET model...")
    birdnet = BirdNETModel(str(MODEL_PATH))

    print(f"\nExtracting train embeddings ({len(train_files)} files)...")
    train_embs = extract_embeddings(birdnet, train_files)

    print(f"\nExtracting test embeddings ({len(test_files)} files)...")
    test_embs  = extract_embeddings(birdnet, test_files)

    embedding_dim = train_embs.shape[1]
    print(f"\nBuilding binary classifier (embedding_dim={embedding_dim})...")
    classifier = build_classifier(embedding_dim).to(device)

    print(f"\nTraining (epochs={N_EPOCHS}, batch={BATCH_SIZE})...")
    classifier = train(classifier, train_embs, train_labels, device)

    print("\nEvaluating on test set...")
    auc, cm = evaluate(classifier, test_embs, test_labels, device)

    figs_path = RESULTS_PATH.parent / "figs"
    plot_confusion_matrix(cm, figs_path / "confusion_matrix_binary_anthro_birdnet.png")

    print("\nSaving classifier bundle...")
    save_classifier_bundle(
        classifier=classifier,
        embedding_model=birdnet,
        model_name="birdnet",
        species=ANTHROPOGENIC + NATURAL,
        training_size=None,
        random_seed=SEED,
        test_auc=auc,
        n_epochs=N_EPOCHS,
        results_path=str(RESULTS_PATH),
        bundle_name="binary_anthro_birdnet",
        labels={
            "0": "natural",
            "1": "anthropogenic",
            "natural_species":       NATURAL,
            "anthropogenic_species": ANTHROPOGENIC,
        },
    )


if __name__ == "__main__":
    main()
