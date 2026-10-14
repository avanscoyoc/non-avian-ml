"""
cross_detection.py

Binary cross-detection: how each class's binary detector responds to every class's test clips.

Two versions (the held-out test recordings are never used for training or for choosing cutoffs):
  default     N = 40 training clips per class (or the largest N a low-sample class reaches): the same clips as
              the multiclass experiment, so binary false alarms and multiclass confusions are like-for-like.
  --all-data  every training clip of the class (1:1 with negatives): the deployment view - the best detector
              this data supports. Rows then differ in training data (device static 1,366 clips ... water 6).

Detectors are trained exactly as in main.py (MLP head, 10 random draws) and run on the test clips of all target
classes. A clip counts as detected when p >= 0.5, the natural cutoff for a head trained on a 1:1 balanced set.
A cell (row i, column j) is the proportion of class-j test clips that detector i calls positive, averaged over
the 10 draws: the diagonal is recall, off-diagonal cells are false alarms. Rows need not sum to 1 (detectors
are independent; one clip can trigger several).

Cutoff check (CSV columns *_tuned): each detector also gets its own cutoff chosen from its training data -
training clips split into folds by recording, each fold scored by a head that never saw those recordings, the
cutoff maximising F1 on those out-of-fold scores (0.5 if the clips come from one recording). In both versions
this raised recall and false alarms together without consistently improving their balance for well-sampled
detectors, and made low-sample detectors worse (folds of 5-10 clips give weak heads and cutoffs that are too
low), so the figures use 0.5.

Outputs: results/experiment/cross_detection[_all_data].csv, figs/cross_detection[_all_data].png/.pdf

Usage (embeddings come from the cache):
    pixi run python experiment/cross_detection.py
    pixi run python experiment/cross_detection.py --all-data
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import FIGS, ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import binary_split, load_audio_files, load_manifest  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.model_loader import torch_device  # noqa: E402
from src.plot_style import MODELS, model_name  # noqa: E402
from src.trainer import predict_heads, train_heads  # noqa: E402
from figures import DIAGONAL_EDGE, NO_DATA, STYLE, _fraction, save  # noqa: E402

HEADLINE_N = 40
THRESHOLD = 0.5  # cutoff for the figures; also the tuned check's fallback
MAX_FOLDS = 5


def oof_threshold(X, rows, labels, groups, cfg, seed) -> float:
    """Cutoff maximising F1 on out-of-fold scores, folds split by recording (training clips only)."""
    rows, y, g = np.asarray(rows), np.asarray(labels), np.asarray(groups)
    k = min(MAX_FOLDS, len(set(g[y == 1])), len(set(g[y == 0])))
    if k < 2:
        return THRESHOLD
    oof = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(rows, y, g)):
        if len(set(y[tr])) < 2:
            continue
        head = train_heads(X, [rows[tr].tolist()], [y[tr].tolist()], batch_size=cfg.batch_size,
                           n_epochs=cfg.n_epochs, seed=seed * 100 + fold)
        oof[va] = predict_heads(head, X, [rows[va].tolist()])[0][:, 1]
    ok = ~np.isnan(oof)
    if len(set(y[ok])) < 2:
        return THRESHOLD
    # Candidates sit midway between consecutive scores; among equally good cutoffs take the middle one, so a
    # detector that separates its training clips perfectly gets the cutoff mid-gap, not at its weakest positive.
    s = np.unique(oof[ok])
    candidates = (s[:-1] + s[1:]) / 2 if len(s) > 1 else s
    f1 = np.array([f1_score(y[ok], oof[ok] >= t) for t in candidates])
    best = candidates[np.isclose(f1, f1.max())]
    return float(best[len(best) // 2])


def run(cfg, all_data: bool) -> pd.DataFrame:
    manifest = load_manifest(cfg.clips_csv)
    partition = build_partition(manifest, cfg)
    splits = {c: binary_split(partition, c, cfg.split_seed) for c in cfg.species}
    classes = [c for c in cfg.species if splits[c].test_pos]
    test_files = [f for c in classes for f in partition.test_clips[c]]
    test_cls = np.array([c for c in classes for _ in partition.test_clips[c]])
    cls_of = dict(zip(manifest.filename, manifest.cls))
    if all_data:
        sizes = {c: splits[c].max_n for c in classes}
    else:
        sizes = {c: min(HEADLINE_N, max([n for n in cfg.training_sizes if n <= splits[c].max_n], default=0))
                 for c in classes}

    draws = {c: [load_audio_files(sizes[c], splits[c].train_pos, splits[c].train_neg, seed) for seed in cfg.random_seeds]
             for c in classes if sizes[c]}
    clips = {(cls_of[f], f) for f in test_files} | {(cls_of[f], f) for d in draws.values() for files, _ in d for f in files}
    device = torch_device()
    rows = []
    for model in [m for m in MODELS if m in cfg.models]:
        table = get_embeddings(model, sorted(clips), cfg.data_path, cfg.embeddings_path)
        X = torch.from_numpy(np.ascontiguousarray(table.emb)).to(device)
        test_rows = table.rows(test_files)
        for c, d in draws.items():
            heads = train_heads(X, [table.rows(f) for f, _ in d], [l for _, l in d], batch_size=cfg.batch_size,
                                n_epochs=cfg.n_epochs, seed=sizes[c])
            cutoffs = [oof_threshold(X, table.rows(f), l, [partition.source[x] for x in f], cfg, seed)
                       for (f, l), seed in zip(d, cfg.random_seeds)]
            probs = predict_heads(heads, X, [test_rows] * len(d))
            detected = np.stack([p[:, 1] >= THRESHOLD for p in probs])
            tuned = np.stack([p[:, 1] >= t for p, t in zip(probs, cutoffs)])
            for target in classes:
                cols = test_cls == target
                rows.append({"model": model, "detector": c, "detector_n": sizes[c], "test_class": target,
                             "test_n": int(cols.sum()), "detection_rate": float(detected[:, cols].mean()),
                             "detection_rate_tuned": float(tuned[:, cols].mean()),
                             "cutoff_tuned_mean": float(np.mean(cutoffs))})
        print(f"[{model}] cross-detection done", flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, classes: list[str], all_data: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    models = [m for m in MODELS if m in set(df.model)]
    n = len(classes)
    n_cols = min(3, len(models))
    n_rows = math.ceil(len(models) / n_cols)
    test_n = df.drop_duplicates("test_class").set_index("test_class").test_n
    det_n = df.drop_duplicates("detector").set_index("detector").detector_n
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 5.4 * n_rows), squeeze=False,
                                 constrained_layout=True)
        axes = axes.flatten()
        for ax in axes[len(models):]:
            ax.set_visible(False)
        for k, (ax, model) in enumerate(zip(axes, models)):
            d = df[df.model == model].pivot(index="detector", columns="test_class", values="detection_rate")
            grid = d.reindex(index=classes, columns=classes).to_numpy()
            im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    if np.isnan(grid[i, j]):
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NO_DATA, edgecolor="#b8bec8",
                                               hatch="///", lw=0))
                        continue
                    ax.text(j, i, _fraction(grid[i, j]), ha="center", va="center", fontsize=5,
                            color="#111111" if grid[i, j] > 0.55 else "white")
                ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DIAGONAL_EDGE, lw=1.5))
            names = [c.replace("_", " ") for c in classes]
            ax.set_xticks(range(n), [f"{nm} ({test_n[c]})" for nm, c in zip(names, classes)], rotation=90, fontsize=6.5)
            ax.set_yticks(range(n), [f"{nm} (N={det_n[c]})" if all_data or det_n.get(c, HEADLINE_N) != HEADLINE_N else nm
                                     for nm, c in zip(names, classes)] if k % n_cols == 0 else [""] * n, fontsize=6.5)
            if k % n_cols == 0:
                ax.set_ylabel("Binary detector (training clips per class)", fontsize=8)
            ax.set_xlabel("Test clips of class (n)", fontsize=8)
            ax.set_title(model_name(model), fontsize=9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#333333")
        trained = "all training clips per class" if all_data else f"N = {HEADLINE_N} training clips per class"
        fig.suptitle(f"Binary cross-detection ({trained}; detected = p >= {THRESHOLD})", fontsize=10)
        fig.colorbar(im, ax=axes[:len(models)].tolist(), shrink=0.5,
                     label="Proportion detected; diagonal = recall, mean of 10 draws")
        save(fig, "cross_detection_all_data" if all_data else "cross_detection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-data", action="store_true", help="Train on every training clip (deployment view).")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(ROOT / "src" / "config.yaml")
    df = run(cfg, args.all_data)
    out = cfg.results_path / "experiment" / f"cross_detection{'_all_data' if args.all_data else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(f"Results: {out}")
    classes = [c for c in cfg.species if c in set(df.detector) | set(df.test_class)]
    plot(df, classes, args.all_data)


if __name__ == "__main__":
    main()
