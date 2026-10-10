"""
zero_shot_label_check.py

Checks the zero-shot label mapping (src/zero_shot_labels.yaml) before the full run, using
TRAINING-POOL clips only (the held-out test recordings are never touched). For each class and
each model with its own labels, writes results/zero_shot_label_check.md with:
  (a) every mapped / candidate label: found in the model's label list or not, and its AUC
      (class vs other classes' training clips);
  (b) the 10 labels the model scores highest on the class's training clips ("what it hears");
  (c) the 10 labels that best separate the class from the rest on training clips (sanity check only;
      picking labels from this list would make the N=0 point no longer zero-shot).

Embeddings/scores come from the cache, so run main.py's extraction first or let this script fill it.

Usage:
    pixi run python scripts/zero_shot_label_check.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import binary_split, load_manifest  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.zero_shot import load_label_map, match_labels, zero_shot_scores  # noqa: E402

LABELLED_MODELS = ["birdnet", "birdnet3", "perch2", "cnn14"]


def column_aucs(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ROC AUC of every column at once (rank formulation, ties averaged)."""
    ranks = rankdata(scores, axis=0)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def label_auc(labels, scores, y, wanted) -> str:
    try:
        idx = match_labels(labels, wanted)
    except KeyError as e:
        return f"**not found** {e}"
    from sklearn.metrics import roc_auc_score
    return f"{roc_auc_score(y, zero_shot_scores(scores, idx)):.3f}"


def main():
    cfg = load_config(ROOT / "src" / "config.yaml")
    manifest = load_manifest(cfg.clips_csv)
    cls_of = dict(zip(manifest.filename, manifest.cls))
    partition = build_partition(manifest, cfg)
    splits = {sp: binary_split(partition, sp, cfg.split_seed) for sp in cfg.species}
    mapping, candidates = load_label_map(), load_label_map(section="candidates")
    models = [m for m in cfg.models if m in LABELLED_MODELS]

    lines = ["# Zero-shot label check (training-pool clips only)", "",
             "AUC = how well the label's score separates the class's training clips from the other classes' "
             "training clips. `mapping` labels are used for the N=0 point; `candidates` are shown for review only.", ""]
    for model in models:
        clips = {(cls_of[f], f) for s in splits.values() for f in s.train_pos + s.train_neg}
        table = get_embeddings(model, sorted(clips), cfg.data_path, cfg.embeddings_path)
        labels = table.labels
        lines += [f"## {model} ({len(labels)} labels)", ""]
        for sp in cfg.species:
            s = splits[sp]
            files = s.train_pos + s.train_neg
            y = np.array([1] * len(s.train_pos) + [0] * len(s.train_neg))
            scores = np.asarray(table.scores)[table.rows(files)]
            lines.append(f"### {sp} — {model}")
            for section, m in (("mapping", mapping), ("candidate", candidates)):
                wanted = m.get(sp, {}).get(model)
                if wanted:
                    lines.append(f"- {section} {wanted}: AUC {label_auc(labels, scores, y, wanted)}")
            if not mapping.get(sp, {}).get(model):
                lines.append("- no mapped label (no N=0 point)")
            mean_pos = scores[y == 1].mean(axis=0)
            top_heard = np.argsort(-mean_pos)[:10]
            aucs = column_aucs(scores, y)
            top_sep = np.argsort(-aucs)[:10]
            lines.append("- top scored on class clips: " + "; ".join(f"{labels[j]} ({aucs[j]:.2f})" for j in top_heard))
            lines.append("- top separating labels: " + "; ".join(f"{labels[j]} ({aucs[j]:.2f})" for j in top_sep))
            lines.append("")

    out = cfg.results_path / "zero_shot_label_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
