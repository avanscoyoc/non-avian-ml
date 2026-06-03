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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score

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
MODEL_PATH  = Path("/workspaces/non-avian-ml/model_birdnet_2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
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
    print(f"\n  AUC: {auc:.4f}")
    print(classification_report(true, preds, target_names=["natural", "anthropogenic"], digits=3))
    return auc


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
    auc = evaluate(classifier, test_embs, test_labels, device)

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
