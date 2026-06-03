"""
train_multiclass_birdnet.py

Trains a single multiclass MLP classifier on BirdNET embeddings.
One class per species, using train/test counts from config.yaml.
No stratified k-fold — a single train/test split.

Usage:
    python scripts/train_multiclass_birdnet.py
"""

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

from src.config import load_config
from src.model_loader import BirdNETModel
from src.results import save_classifier_bundle


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collect_split(data_path: str, species: list[str], species_sizes: dict, datatype: str, seed: int):
    """
    For each species, select positive files for train and test using the
    sizes specified in species_sizes.  Files are drawn from:
        <data_path>/<species>/<datatype>/pos/

    Returns:
        train_files  : list of file paths
        train_labels : list of int class indices
        test_files   : list of file paths
        test_labels  : list of int class indices
        label_names  : list of species names (index == class label)
    """
    train_files, train_labels = [], []
    test_files,  test_labels  = [], []
    label_names = []

    for class_idx, sp in enumerate(species):
        pos_dir = Path(data_path) / sp / datatype / "pos"
        all_pos = sorted(pos_dir.glob("*.wav"))

        if species_sizes and sp in species_sizes:
            n_train = species_sizes[sp]["train"]
            n_test  = species_sizes[sp]["test"]
        else:
            raise ValueError(f"No species_sizes entry for '{sp}'")

        total_needed = n_train + n_test
        if len(all_pos) < total_needed:
            raise ValueError(
                f"{sp}: need {total_needed} pos files "
                f"(train={n_train} + test={n_test}), "
                f"found {len(all_pos)}"
            )

        rng = random.Random(seed)
        shuffled = list(all_pos)
        rng.shuffle(shuffled)

        test_batch  = [str(f) for f in shuffled[:n_test]]
        train_batch = [str(f) for f in shuffled[n_test : n_test + n_train]]

        test_files  .extend(test_batch)
        test_labels .extend([class_idx] * len(test_batch))
        train_files .extend(train_batch)
        train_labels.extend([class_idx] * len(train_batch))
        label_names.append(sp)

    return train_files, train_labels, test_files, test_labels, label_names


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_all_embeddings(model: BirdNETModel, files: list[str]) -> torch.Tensor:
    embeddings = []
    for i, fp in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(files)} files processed")
        emb = model.extract_embeddings(fp)
        embeddings.append(emb.flatten())
    return torch.stack(embeddings)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_classifier(embedding_dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    labels: list[int],
    device: torch.device,
    batch_size: int,
    n_epochs: int,
) -> nn.Module:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(embeddings.to(device), labels_tensor.to(device))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Inverse-frequency class weights so minority classes aren't drowned out
    counts = torch.bincount(labels_tensor, minlength=labels_tensor.max().item() + 1).float()
    class_weights = (1.0 / counts.clamp(min=1)).to(device)
    class_weights = class_weights / class_weights.sum() * len(counts)  # normalise

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    classifier.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for feats, lbls in loader:
            optimizer.zero_grad()
            loss = criterion(classifier(feats), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch + 1}/{n_epochs}  loss={epoch_loss / len(loader):.4f}")

    return classifier


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    labels: list[int],
    label_names: list[str],
    device: torch.device,
) -> dict:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.to(device))
        preds  = torch.argmax(logits, dim=1).cpu().numpy()

    true = np.array(labels)
    acc  = accuracy_score(true, preds)

    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    ovr_auc = roc_auc_score(true, probs, multi_class="ovr", average="macro")

    print(f"\n  Overall accuracy: {acc:.4f}  |  Macro OVR AUC: {ovr_auc:.4f}")
    print("\n  Per-class report:")
    print(classification_report(true, preds, target_names=label_names, digits=3))

    cm = confusion_matrix(true, preds)
    return {"accuracy": acc, "ovr_auc": ovr_auc, "predictions": preds, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, label_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(label_names)), max(6, len(label_names) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Multiclass Confusion Matrix (BirdNET)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = Path(__file__).parent.parent / "src" / "config.yaml"
    config = load_config(str(config_path))

    birdnet_model_path = str(
        Path(__file__).parent.parent
        / "models"
        / "birdnet_2.4"
        / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use the first random seed from config (no multiple seeds needed here)
    seed = config.random_seeds[0] if config.random_seeds else 42

    print(f"\nCollecting files for {len(config.species)} species (seed={seed})...")
    train_files, train_labels, test_files, test_labels, label_names = collect_split(
        data_path=config.data_path,
        species=config.species,
        species_sizes=config.species_sizes,
        datatype=config.datatype,
        seed=seed,
    )
    print(f"  Train: {len(train_files)} files | Test: {len(test_files)} files")

    print("\nLoading BirdNET model...")
    birdnet = BirdNETModel(birdnet_model_path)

    print(f"\nExtracting train embeddings ({len(train_files)} files)...")
    train_embs = extract_all_embeddings(birdnet, train_files)

    print(f"\nExtracting test embeddings ({len(test_files)} files)...")
    test_embs = extract_all_embeddings(birdnet, test_files)

    n_classes     = len(label_names)
    embedding_dim = train_embs.shape[1]
    print(f"\nBuilding {n_classes}-class classifier (embedding_dim={embedding_dim})...")
    classifier = build_classifier(embedding_dim, n_classes).to(device)

    print(f"\nTraining (epochs={config.n_epochs}, batch={config.batch_size})...")
    classifier = train_classifier(
        classifier, train_embs, train_labels,
        device, config.batch_size, config.n_epochs,
    )

    print("\nEvaluating on test set...")
    results = evaluate_classifier(classifier, test_embs, test_labels, label_names, device)

    figs_path = Path(__file__).parent.parent / "figs"
    plot_confusion_matrix(
        results["confusion_matrix"],
        label_names,
        figs_path / "confusion_matrix_multiclass_birdnet.png",
    )

    print("\nSaving classifier bundle...")
    save_classifier_bundle(
        classifier=classifier,
        embedding_model=birdnet,
        model_name="birdnet",
        species=label_names,
        training_size=None,
        random_seed=seed,
        test_auc=results["ovr_auc"],
        n_epochs=config.n_epochs,
        results_path=config.results_path,
        bundle_name="multiclass_birdnet",
        labels={str(i): sp for i, sp in enumerate(label_names)},
    )


if __name__ == "__main__":
    main()
