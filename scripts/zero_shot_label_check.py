"""Checks the zero-shot label mapping (`zero_shot_labels` in config.yaml) before the saturation run, using
TRAINING-POOL clips only (the held-out test recordings are never touched). For each class and each model with its
own labels, writes results/zero_shot_label_check.md with:
  (a) every mapped label: found in the model's label list or not, and its AUC
      (class vs other classes' training clips);
  (b) the 10 labels the model scores highest on the class's training clips ("what it hears");
  (c) the 10 labels that best separate the class from the rest on training clips (sanity check only;
      picking labels from this list would make the N=0 point no longer zero-shot).

Embeddings and label scores come from the cache (and fill it if needed).

Usage:
    pixi run zero-shot-labels
"""
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from core.config import load_config
from core.paths import CONFIG
from core.study import load_study
from core.zero_shot import match_labels, zero_shot_scores

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
    return f"{roc_auc_score(y, zero_shot_scores(scores, idx)):.3f}"


def main():
    parser = argparse.ArgumentParser(description="Check the zero-shot label mapping on training clips.")
    parser.add_argument("--config", default=CONFIG)
    cfg = load_config(parser.parse_args().config)
    study = load_study(cfg)
    mapping = cfg.zero_shot_labels
    models = [m for m in cfg.models if m in LABELLED_MODELS]

    lines = ["# Zero-shot label check (training-pool clips only)", "",
             "AUC = how well the label's score separates the class's training clips from the other classes' "
             "training clips. Mapped labels give the N=0 point.", ""]
    for model in models:
        table, _ = study.embed(model, [f for s in study.splits.values() for f in s.train_pos + s.train_neg])
        labels = table.labels
        lines += [f"## {model} ({len(labels)} labels)", ""]
        for c in cfg.classes:
            s = study.splits[c]
            files = s.train_pos + s.train_neg
            y = np.array([1] * len(s.train_pos) + [0] * len(s.train_neg))
            scores = np.asarray(table.scores)[table.rows(files)]
            lines.append(f"### {c} — {model}")
            wanted = mapping.get(c, {}).get(model)
            if wanted:
                lines.append(f"- mapping {wanted}: AUC {label_auc(labels, scores, y, wanted)}")
            else:
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
