"""True zero-shot: score test clips with the model's own label(s) for the class, no training.

The class -> label mapping lives in core/zero_shot_labels.yaml and should be reviewed with
scripts/zero_shot_label_check.py (training clips only) before the full run.
"""
import numpy as np
import yaml

from core.paths import ROOT

LABEL_MAP_PATH = ROOT / "core" / "zero_shot_labels.yaml"


def load_label_map(path=LABEL_MAP_PATH, section="mapping") -> dict[str, dict[str, list[str]]]:
    """{class: {model: [label, ...]}} from the YAML's `mapping` (used) or `candidates` (report only)."""
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get(section) or {}


def match_labels(model_labels: list[str], wanted: list[str]) -> list[int]:
    """Indices of the wanted labels. A label matches its full string or its scientific-name part
    (before "_"); "Genus *" matches every species of that genus."""
    lookup = {}
    for i, label in enumerate(model_labels):
        lookup.setdefault(label, i)
        lookup.setdefault(label.split("_")[0], i)
    idx, missing = set(), []
    for w in wanted:
        if w.endswith(" *"):
            hits = {i for i, label in enumerate(model_labels) if label.startswith(w[:-1])}
        else:
            hits = {lookup[w]} if w in lookup else set()
        idx |= hits
        if not hits:
            missing.append(w)
    if missing:
        raise KeyError(f"labels not found in model label list: {missing}")
    return sorted(idx)


def zero_shot_scores(scores: np.ndarray, label_idx: list[int]) -> np.ndarray:
    """Max score over the mapped labels (raw logits or probabilities; AUC is unaffected)."""
    return np.asarray(scores[:, label_idx]).max(axis=1)
