"""One multiclass classifier ("which noise is this?"), Figures 4-6 of docs/brief.md.

A single MLP head over the target classes with at least headline_n (40) training clips; the others are listed as
waiting for data and join automatically once data/metadata.csv has enough. Training, per seed (1-10): exactly the
N positive clips per class that the binary N-clip draw used for that class and seed. Test set: each class's
held-out clips from the shared split, the same clips as its binary test positives. Classes with no test clips get
NaN metrics and a hatched confusion-matrix row instead of an error.

Outputs in results/multiclass/:
  {model}.csv                   per seed x class: test n, one-vs-rest AUC, precision, recall, F1, macro F1
  confusion_{model}.npy         confusion matrices (counts), shape (seeds, classes, classes)
  test_embeddings_{model}.npz   test-clip embeddings + labels, for the t-SNE figure
  classes.json                  class names (index order) and test clips per class
  summary.csv                   mean and SD over seeds per model x class
Figures (after experiments.saturation, for the binary AUCs): binary_vs_multiclass_auc (Figure 6),
confusion_matrices (Figure 4), tsne_embeddings (Figure 5).

Usage:
    pixi run multiclass                                  # fit the models not done yet, then plot
    pixi run python -m experiments.multiclass --models birdnet perch2
    pixi run python -m experiments.multiclass --plot
"""
import argparse
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from core.config import load_config
from core.encoders import MODEL_NAMES
from core.heads import predict_heads, train_heads
from core.paths import CONFIG
from core.plotting import CLASS_COLORS, MODELS, PANEL_STYLE, class_name, draw_matrix, model_name, panel_grid, save
from core.study import load_study


def positives(split, n: int, seed: int) -> list[str]:
    """The n positive clips the binary n-clip draw used for this class and seed."""
    files, labels = split.draw(n, seed)
    return sorted(f for f, label in zip(files, labels) if label == 1)


def evaluate(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    """Metrics that stay defined when a class has no test clips (its row/metrics become empty/NaN)."""
    preds = probs.argmax(axis=1)
    present = sorted(set(y.tolist()))
    cm = confusion_matrix(y, preds, labels=list(range(n_classes)))
    auc = [roc_auc_score(y == i, probs[:, i]) if i in present and len(present) > 1 else np.nan
           for i in range(n_classes)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, labels=list(range(n_classes)), zero_division=0)
    absent = ~np.isin(np.arange(n_classes), present)
    precision, recall, f1 = (np.where(absent, np.nan, a) for a in (precision, recall, f1))
    return {"cm": cm, "auc": auc, "precision": precision, "recall": recall, "f1": f1,
            "macro_f1": float(np.nanmean(f1))}


def design(cfg, study):
    """The clips: classes with at least headline_n (N) training clips; their test clips and class indices; and per
    seed one training set, each class's N positive clips from its binary N-clip draw. Returns (classes, classes
    waiting for data with their training clip counts, test files, test class indices, training sets)."""
    waiting = {c: len(s.train_pos) for c, s in study.splits.items() if len(s.train_pos) < cfg.headline_n}
    classes = [c for c in cfg.classes if c not in waiting]
    test_files = [f for c in classes for f in study.partition.test_clips[c]]
    y_test = np.array([i for i, c in enumerate(classes) for _ in study.partition.test_clips[c]])
    n = cfg.headline_n
    draws = [([f for c in classes for f in positives(study.splits[c], n, seed)],
              [i for i, c in enumerate(classes) for _ in range(n)]) for seed in cfg.random_seeds]
    return classes, waiting, test_files, y_test, draws


def run_model(model, classes, test_files, y_test, draws, study, cfg, out):
    table, X = study.embed(model, test_files + [f for files, _ in draws for f in files])
    test_rows = table.rows(test_files)

    heads = train_heads(X, [table.rows(f) for f, _ in draws], [l for _, l in draws], n_classes=len(classes),
                        batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, seed=cfg.headline_n)
    rows, cms = [], []
    for seed, probs in zip(cfg.random_seeds, predict_heads(heads, X, test_rows)):
        m = evaluate(probs, y_test, len(classes))
        cms.append(m["cm"])
        rows += [{"model": model, "seed": seed, "class": c, "class_idx": i,
                  "test_n": int((y_test == i).sum()), "ovr_auc": m["auc"][i], "precision": m["precision"][i],
                  "recall": m["recall"][i], "f1": m["f1"][i], "macro_f1": m["macro_f1"]}
                 for i, c in enumerate(classes)]

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{model}.csv", index=False)
    np.save(out / f"confusion_{model}.npy", np.stack(cms))
    np.savez(out / f"test_embeddings_{model}.npz", embeddings=table.emb[test_rows],
             labels=y_test, class_names=np.array(classes))
    print(f"  {model:8s} macro F1 {df.groupby('seed').macro_f1.first().mean():.3f} | "
          f"OvR AUC " + ", ".join(f"{c} {df[df['class'] == c].ovr_auc.mean():.3f}" for c in classes))


def run(cfg, models, out):
    study = load_study(cfg)
    classes, waiting, test_files, y_test, draws = design(cfg, study)
    if waiting:
        print(f"Waiting for more data (fewer than {cfg.headline_n} training clips): {waiting}")
    test_n = [len(study.partition.test_clips[c]) for c in classes]
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps({"classes": classes, "test_n": test_n}))
    print(f"Test clips per class {dict(zip(classes, test_n))}")
    for model in models:
        if (out / f"{model}.csv").exists():
            print(f"  {model}: done (delete results/multiclass/{model}.csv to rerun)")
            continue
        run_model(model, classes, test_files, y_test, draws, study, cfg, out)

    dfs = [pd.read_csv(out / f"{m}.csv") for m in MODEL_NAMES if (out / f"{m}.csv").exists()]
    agg = pd.concat(dfs).groupby(["model", "class"])[["ovr_auc", "f1", "macro_f1"]].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(out / "summary.csv", index=False)
    print(f"\nAggregated results -> {out / 'summary.csv'}")


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_binary_vs_multiclass(cfg, out, classes):
    """Figure 6: binary AUC at the headline N vs multiclass one-vs-rest AUC, same positive test clips."""
    n = cfg.headline_n
    binary = pd.read_csv(cfg.results_path / "saturation" / "auc_by_n.csv")
    binary = binary[binary["class"].isin(classes) & (binary.training_size == n)][["class", "model", "test_auc_mean"]]
    binary = binary.rename(columns={"test_auc_mean": "binary_auc"})
    multi = pd.read_csv(out / "summary.csv")[["model", "class", "ovr_auc_mean"]]
    both = binary.merge(multi, on=["model", "class"])
    models = [m for m in MODELS if m in set(both.model)]
    lo = math.floor(min(both.binary_auc.min(), both.ovr_auc_mean.min()) * 20) / 20

    with plt.rc_context(PANEL_STYLE):
        fig, axes = panel_grid(len(models), 3.4, 3.4)
        for ax, model in zip(axes, models):
            d = both[both.model == model]
            ax.plot([lo, 1], [lo, 1], "--", color="#bbbbbb", lw=0.9, zorder=0)
            for i, c in enumerate(classes):
                r = d[d["class"] == c]
                if len(r):
                    ax.scatter(r.binary_auc, r.ovr_auc_mean, color=CLASS_COLORS[i], s=40, zorder=3,
                               edgecolors="white", linewidths=0.5, label=class_name(c))
            ax.set_xlim(lo, 1.005); ax.set_ylim(lo, 1.005); ax.set_aspect("equal")
            ax.set_title(model_name(model), fontweight="bold")
            ax.set_xlabel(f"Binary AUC (N={n})"); ax.set_ylabel("Multiclass one-vs-rest AUC")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4), frameon=False)
        save(fig, cfg.figs_path / "binary_vs_multiclass_auc")


def fig_confusion_matrices(cfg, out, classes, test_n):
    """Figure 4: row-normalised confusion matrices (counts pooled over the draws; rows sum to 1)."""
    models = [m for m in MODELS if (out / f"confusion_{m}.npy").exists()]
    names = [class_name(c) for c in classes]
    with plt.rc_context(PANEL_STYLE):
        fig, axes = panel_grid(len(models), 3.6, 3.6)
        for k, (ax, model) in enumerate(zip(axes, models)):
            cm = np.load(out / f"confusion_{model}.npy").sum(axis=0)
            rows = cm.sum(axis=1, keepdims=True)
            frac = np.divide(cm, rows, out=np.full(cm.shape, np.nan), where=rows > 0)
            im = draw_matrix(ax, frac, fontsize=6.5, diagonal_lw=1.8)
            ax.set_xticks(range(len(classes)), names, rotation=90, fontsize=7)
            ax.set_yticks(range(len(classes)), [f"{nm} (n={t})" for nm, t in zip(names, test_n)] if k % 3 == 0
                          else [""] * len(classes), fontsize=7)
            ax.set_xlabel("Predicted class", fontsize=8)
            if k % 3 == 0:
                ax.set_ylabel("True class (test clips)", fontsize=8)
            ax.set_title(model_name(model), fontsize=9)
        fig.colorbar(im, ax=axes[:len(models)].tolist(), shrink=0.6,
                     label="Proportion of a class's test clips (diagonal = recall), 10 draws pooled")
        save(fig, cfg.figs_path / "confusion_matrices")


def fig_tsne(cfg, out, classes, perplexity: int = 30, seed: int = 0):
    """Figure 5: t-SNE of the frozen test-clip embeddings; stars mark class centroids."""
    models = [m for m in MODELS if (out / f"test_embeddings_{m}.npz").exists()]
    with plt.rc_context(PANEL_STYLE):
        fig, axes = panel_grid(len(models), 3.4, 3.4)
        for ax, model in zip(axes, models):
            data = np.load(out / f"test_embeddings_{model}.npz")
            embs, labels = data["embeddings"], data["labels"]
            print(f"  t-SNE {model} {embs.shape}")
            xy = TSNE(n_components=2, perplexity=min(perplexity, len(embs) - 1), random_state=seed,
                      max_iter=1000).fit_transform(embs)
            for i, c in enumerate(classes):
                pts = xy[labels == i]
                if len(pts):
                    ax.scatter(pts[:, 0], pts[:, 1], color=CLASS_COLORS[i], s=10, alpha=0.7, linewidths=0,
                               label=class_name(c))
                    ax.scatter(*pts.mean(axis=0), color=CLASS_COLORS[i], marker="*", s=140,
                               edgecolors="white", linewidths=0.8, zorder=5)
            ax.set_title(model_name(model), fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 4), frameon=False,
                   markerscale=2, title="★ = class centroid")
        save(fig, cfg.figs_path / "tsne_embeddings")


def plot(cfg, out):
    info = json.loads((out / "classes.json").read_text())
    fig_binary_vs_multiclass(cfg, out, info["classes"])
    fig_confusion_matrices(cfg, out, info["classes"], info["test_n"])
    fig_tsne(cfg, out, info["classes"])


def main():
    parser = argparse.ArgumentParser(description="Multiclass head over the noise classes.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="Default: all models in the config.")
    parser.add_argument("--plot", action="store_true", help="Only redraw the figures from saved results.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(args.config)
    out = cfg.results_path / "multiclass"
    if not args.plot:
        run(cfg, args.models or cfg.models, out)
    plot(cfg, out)


if __name__ == "__main__":
    main()
