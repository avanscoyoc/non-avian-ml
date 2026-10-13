"""
head_comparison.py

Robustness check: does the choice of classifier head change the results? The main experiment trains a small
MLP head; here a linear probe (standardised logistic regression, C = 1) is trained on exactly the same cached
embeddings, training draws and test clips, and compared with the MLP results from main.py.

Outputs: results/head_comparison.csv (per class x model x N: both heads' mean test AUC and 95% CI over the
10 draws) and figs/head_comparison.png/.pdf (mean over well-sampled classes, one panel per model).

Usage (after main.py):
    pixi run python scripts/head_comparison.py
"""
import sys
import warnings
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import binary_split, load_audio_files, load_manifest  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.plot_style import AXIS, GRID, INK, INK_SECONDARY, MODELS, model_name  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def linear_probe_auc(E, train_rows, train_labels, test_rows, test_labels) -> float:
    scaler = StandardScaler().fit(E[train_rows])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(E[train_rows]), train_labels)
    return roc_auc_score(test_labels, clf.decision_function(scaler.transform(E[test_rows])))


def run(cfg) -> pd.DataFrame:
    manifest = load_manifest(cfg.clips_csv)
    partition = build_partition(manifest, cfg)
    splits = {c: binary_split(partition, c, cfg.split_seed) for c in cfg.species}
    cls_of = dict(zip(manifest.filename, manifest.cls))
    clips = sorted({(cls_of[f], f) for s in splits.values() for f in s.train_pos + s.train_neg + s.test_pos + s.test_neg})
    rows = []
    for model in cfg.models:
        table = get_embeddings(model, clips, cfg.data_path, cfg.embeddings_path)
        E = np.ascontiguousarray(table.emb)
        for c, s in splits.items():
            mlp = pd.read_csv(cfg.results_path / f"results_{c}.csv")
            mlp = mlp[mlp.model == model].set_index("training_size")
            test_rows = table.rows(s.test_pos + s.test_neg)
            test_labels = [1] * len(s.test_pos) + [0] * len(s.test_neg)
            for n in cfg.training_sizes:
                if n > s.max_n:
                    continue
                aucs = [linear_probe_auc(E, table.rows(f), l, test_rows, test_labels)
                        for f, l in (load_audio_files(n, s.train_pos, s.train_neg, seed) for seed in cfg.random_seeds)]
                rows.append({"class": c, "model": model, "training_size": n, "low_sample": c in partition.low_sample,
                             "linear_auc_mean": np.mean(aucs), "linear_ci_95": 1.96 * np.std(aucs, ddof=1) / np.sqrt(len(aucs)),
                             "mlp_auc_mean": mlp.loc[n, "test_auc_mean"], "mlp_ci_95": mlp.loc[n, "test_auc_ci_95"]})
        print(f"[{model}] linear probe done", flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = df[~df.low_sample].groupby(["model", "training_size"], as_index=False)[["linear_auc_mean", "mlp_auc_mean"]].mean()
    models = [m for m in MODELS if m in set(w.model)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes.flatten(), models):
        d = w[w.model == model]
        color = MODELS[model][1]
        x = [log2(n) for n in d.training_size]
        ax.plot(x, d.mlp_auc_mean, color=color, marker=MODELS[model][2], linewidth=2, label="MLP head (main)")
        ax.plot(x, d.linear_auc_mean, color=color, marker=MODELS[model][2], linewidth=2, linestyle="--",
                mfc="white", label="Linear probe")
        ax.set_title(model_name(model), color=INK, fontweight="semibold")
        ax.set_xticks(x, [str(n) for n in d.training_size])
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.supxlabel("Training size (clips per class, log scale)", color=INK)
    fig.supylabel("Mean test ROC-AUC (well-sampled classes)", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"head_comparison.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGS / 'head_comparison'}.{{png,pdf}}")


def main():
    cfg = load_config(ROOT / "src" / "config.yaml")
    df = run(cfg)
    df.round(4).to_csv(cfg.results_path / "head_comparison.csv", index=False)
    w = df[~df.low_sample]
    rank = w.groupby("model")[["linear_auc_mean", "mlp_auc_mean"]].mean()
    print("\nMean test AUC over well-sampled classes and all N:")
    print(rank.assign(diff=rank.mlp_auc_mean - rank.linear_auc_mean).sort_values("mlp_auc_mean", ascending=False).round(3).to_string())
    print("model ranking identical:",
          list(rank.mlp_auc_mean.sort_values().index) == list(rank.linear_auc_mean.sort_values().index))
    plot(df)


if __name__ == "__main__":
    main()
