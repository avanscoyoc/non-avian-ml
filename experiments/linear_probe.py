"""Robustness check, Figure S2 of docs/brief.md: does the classifier head change the saturation curves?

A linear probe (standardised logistic regression, C = 1, fitted to convergence) is trained on exactly the same
cached embeddings, training draws and test clips as the MLP heads of experiments.saturation (run that first) and
compared with its results.

Outputs: results/linear_probe/auc_by_n.csv (per class x model x N: both heads' mean test AUC and 95% CI over the
10 draws). Figure: head_comparison (mean over the well-sampled classes, one panel per model).

Usage:
    pixi run linear-probe                                  # fit (unless the CSV exists), then plot
    pixi run python -m experiments.linear_probe --plot     # figure only
"""
import argparse
import warnings
from math import log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from core.config import load_config
from core.paths import CONFIG
from core.plotting import INK, MODELS, model_name, save, style_axes
from core.study import load_study

warnings.filterwarnings("ignore", category=UserWarning)


def linear_probe_auc(E, train_rows, train_labels, test_rows, test_labels) -> float:
    scaler = StandardScaler().fit(E[train_rows])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(E[train_rows]), train_labels)
    return roc_auc_score(test_labels, clf.decision_function(scaler.transform(E[test_rows])))


def run(cfg) -> pd.DataFrame:
    study = load_study(cfg)
    auc = pd.read_csv(cfg.results_path / "saturation" / "auc_by_n.csv")
    files = [f for s in study.splits.values() for f in s.files]
    rows = []
    for model in cfg.models:
        table, _ = study.embed(model, files)
        E = np.ascontiguousarray(table.emb)
        for c, s in study.splits.items():
            mlp = auc[(auc["class"] == c) & (auc.model == model)].set_index("training_size")
            test_rows = table.rows(s.test_files)
            for n in cfg.training_sizes:
                if n > s.max_n:
                    continue
                aucs = [linear_probe_auc(E, table.rows(f), l, test_rows, s.test_labels)
                        for f, l in s.draws(n, cfg.random_seeds)]
                rows.append({"class": c, "model": model, "training_size": n,
                             "low_sample": c in study.partition.low_sample,
                             "linear_auc_mean": np.mean(aucs), "linear_ci_95": 1.96 * np.std(aucs, ddof=1) / np.sqrt(len(aucs)),
                             "mlp_auc_mean": mlp.loc[n, "test_auc_mean"], "mlp_ci_95": mlp.loc[n, "test_auc_ci_95"]})
        print(f"[{model}] linear probe done", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame):
    rank = df[~df.low_sample].groupby("model")[["linear_auc_mean", "mlp_auc_mean"]].mean()
    print("\nMean test AUC over well-sampled classes and all N:")
    print(rank.assign(diff=rank.mlp_auc_mean - rank.linear_auc_mean)
          .sort_values("mlp_auc_mean", ascending=False).round(3).to_string())
    print("model ranking identical:",
          list(rank.mlp_auc_mean.sort_values().index) == list(rank.linear_auc_mean.sort_values().index))


def plot(cfg, df: pd.DataFrame):
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
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.supxlabel("Training size (clips per class, log scale)", color=INK)
    fig.supylabel("Mean test ROC-AUC (well-sampled classes)", color=INK)
    save(fig, cfg.figs_path / "head_comparison", facecolor="white")


def main():
    parser = argparse.ArgumentParser(description="Linear probe vs MLP head.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--plot", action="store_true", help="Only redraw the figure from saved results.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg.results_path / "linear_probe" / "auc_by_n.csv"
    if not args.plot:
        if out.exists():
            print(f"{out} exists (delete it to refit)")
        else:
            df = run(cfg)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.round(4).to_csv(out, index=False)
            report(df)
    plot(cfg, pd.read_csv(out))


if __name__ == "__main__":
    main()
