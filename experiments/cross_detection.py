"""Binary cross-detection, Figures 3 and S3 and the supplementary note on cutoffs in docs/brief.md: how each
class's binary detector responds to every class's test clips.

Two versions, both run each time; neither uses the held-out test recordings for training or for choosing cutoffs:
  headline    N = headline_n (40) training clips per class, or the largest N a low-sample class reaches: the same
              clips as the multiclass experiment, so binary false alarms and multiclass confusions are like-for-like.
  all_data    every training clip of the class (1:1 with negatives): the deployment view, the best detector this
              data supports. Rows then differ in training data (device static 1,366 clips ... water 6).

Detectors are trained as in the saturation experiment (MLP head, 10 random draws) and run on the test clips of all
target classes. A clip counts as detected when p >= 0.5, the natural cutoff for a head trained on a 1:1 balanced
set. A cell (row i, column j) is the proportion of class-j test clips that detector i calls positive, averaged
over the 10 draws: the diagonal is recall, off-diagonal cells are false alarms. Rows need not sum to 1 (detectors
are independent; one clip can trigger several).

Cutoff check (CSV columns *_tuned): each detector also gets its own cutoff chosen from its training data -
training clips split into folds by recording, each fold scored by a head that never saw those recordings, the
cutoff maximising F1 on those out-of-fold scores (0.5 if the clips come from one recording). In both versions
this raised recall and false alarms together without consistently improving their balance for well-sampled
detectors, and made low-sample detectors worse (folds of 5-10 clips give weak heads and cutoffs that are too
low), so the figures use 0.5.

Outputs: results/cross_detection/{headline,all_data}.csv. Figures: cross_detection, cross_detection_all_data.

Usage:
    pixi run cross-detection                                # both versions (each skipped if its CSV exists), then plot
    pixi run python -m experiments.cross_detection --plot   # figures only
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from core.config import load_config
from core.heads import predict_heads, train_heads
from core.paths import CONFIG
from core.plotting import MODELS, PANEL_STYLE, class_name, draw_matrix, model_name, panel_grid, save
from core.study import load_study

THRESHOLD = 0.5  # cutoff for the figures; also the tuned check's fallback
MAX_FOLDS = 5
VERSIONS = {"headline": False, "all_data": True}  # results/cross_detection/{name}.csv -> trained on all clips?


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
        oof[va] = predict_heads(head, X, rows[va].tolist())[0, :, 1]
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
    study = load_study(cfg)
    splits, partition = study.splits, study.partition
    classes = [c for c in cfg.classes if splits[c].test_pos]
    test_files = [f for c in classes for f in partition.test_clips[c]]
    test_cls = np.array([c for c in classes for _ in partition.test_clips[c]])
    if all_data:
        sizes = {c: splits[c].max_n for c in classes}
    else:
        sizes = {c: min(cfg.headline_n, max([n for n in cfg.training_sizes if n <= splits[c].max_n], default=0))
                 for c in classes}

    draws = {c: splits[c].draws(sizes[c], cfg.random_seeds) for c in classes if sizes[c]}
    files = test_files + [f for d in draws.values() for draw_files, _ in d for f in draw_files]
    rows = []
    for model in [m for m in MODELS if m in cfg.models]:
        table, X = study.embed(model, files)
        test_rows = table.rows(test_files)
        for c, d in draws.items():
            heads = train_heads(X, [table.rows(f) for f, _ in d], [l for _, l in d], batch_size=cfg.batch_size,
                                n_epochs=cfg.n_epochs, seed=sizes[c])
            cutoffs = [oof_threshold(X, table.rows(f), l, [partition.source[x] for x in f], cfg, seed)
                       for (f, l), seed in zip(d, cfg.random_seeds)]
            probs = predict_heads(heads, X, test_rows)[:, :, 1]
            detected = probs >= THRESHOLD
            tuned = np.stack([p >= t for p, t in zip(probs, cutoffs)])
            for target in classes:
                cols = test_cls == target
                rows.append({"model": model, "detector": c, "detector_n": sizes[c], "test_class": target,
                             "test_n": int(cols.sum()), "detection_rate": float(detected[:, cols].mean()),
                             "detection_rate_tuned": float(tuned[:, cols].mean()),
                             "cutoff_tuned_mean": float(np.mean(cutoffs))})
        print(f"[{model}] cross-detection done", flush=True)
    return pd.DataFrame(rows)


def plot(cfg, df: pd.DataFrame, all_data: bool):
    classes = [c for c in cfg.classes if c in set(df.detector) | set(df.test_class)]
    models = [m for m in MODELS if m in set(df.model)]
    n = len(classes)
    n_cols = min(3, len(models))
    test_n = df.drop_duplicates("test_class").set_index("test_class").test_n
    det_n = df.drop_duplicates("detector").set_index("detector").detector_n
    names = [class_name(c) for c in classes]
    with plt.rc_context(PANEL_STYLE):
        fig, axes = panel_grid(len(models), 5.6, 5.4)
        for k, (ax, model) in enumerate(zip(axes, models)):
            d = df[df.model == model].pivot(index="detector", columns="test_class", values="detection_rate")
            im = draw_matrix(ax, d.reindex(index=classes, columns=classes).to_numpy(), fontsize=5, diagonal_lw=1.5)
            ax.set_xticks(range(n), [f"{nm} ({test_n[c]})" for nm, c in zip(names, classes)], rotation=90, fontsize=6.5)
            ax.set_yticks(range(n), [f"{nm} (N={det_n[c]})" if all_data or det_n.get(c, cfg.headline_n) != cfg.headline_n
                                     else nm for nm, c in zip(names, classes)] if k % n_cols == 0 else [""] * n,
                          fontsize=6.5)
            if k % n_cols == 0:
                ax.set_ylabel("Binary detector (training clips per class)", fontsize=8)
            ax.set_xlabel("Test clips of class (n)", fontsize=8)
            ax.set_title(model_name(model), fontsize=9)
        trained = "all training clips per class" if all_data else f"N = {cfg.headline_n} training clips per class"
        fig.suptitle(f"Binary cross-detection ({trained}; detected = p >= {THRESHOLD})", fontsize=10)
        fig.colorbar(im, ax=axes[:len(models)].tolist(), shrink=0.5,
                     label="Proportion detected; diagonal = recall, mean of 10 draws")
        save(fig, cfg.figs_path / ("cross_detection_all_data" if all_data else "cross_detection"))


def main():
    parser = argparse.ArgumentParser(description="Binary cross-detection.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--plot", action="store_true", help="Only redraw the figures from saved results.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(args.config)
    for name, all_data in VERSIONS.items():
        out = cfg.results_path / "cross_detection" / f"{name}.csv"
        if not args.plot:
            if out.exists():
                print(f"{out} exists (delete it to rerun)")
            else:
                df = run(cfg, all_data)
                out.parent.mkdir(parents=True, exist_ok=True)
                df.round(4).to_csv(out, index=False)
                print(f"Results: {out}")
        plot(cfg, pd.read_csv(out), all_data)


if __name__ == "__main__":
    main()
