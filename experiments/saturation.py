"""Training-size (saturation) experiment: Figures 1, 2 and S1 of docs/brief.md and the known-vs-new numbers.

For each target class and model, binary MLP heads are trained on frozen embeddings with N clips of the class and
N negatives (N = 5 ... 160 as far as the class's training pool allows, 10 random draws each) and scored on the
class's fixed test clips. N = 0 is zero-shot: the model's own label for the class, where it has one
(`zero_shot_labels` in config.yaml). Classes with such a label are "known" to the model, the rest are "new".

Outputs in results/saturation/:
  runs/{class}_{model}.csv             test AUC of every draw (written last: marks the class x model as done)
  predictions/{class}_{model}.csv.gz   every test clip's score, for bootstraps or paired model comparisons
  auc_by_n.csv                         mean, SD and 95% CI over the draws per class x model x N
  summary.csv, summary.md              per class x model: label status, zero-shot AUC, AUC at the headline N,
                                       best AUC and N, plateau N, break-even N, outcome
Figures: saturation_main and saturation_low_sample (curves), saturation_summary (heatmap at the headline N).

Usage:
    pixi run saturation                                 # train what's missing, summarise, plot
    pixi run python -m experiments.saturation --plot    # figures only
    # On a GPU server: TensorFlow models (CPU) and PyTorch models (GPU) as two parallel jobs
    pixi run -e gpu python -m experiments.saturation --models birdnet perch perch2
    pixi run -e gpu python -m experiments.saturation --models birdnet3 cnn14 resnet
"""
import argparse
import math
import sys
import textwrap
import time
from math import log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from core.binary import aggregate, load_runs, run_class, save_csv
from core.config import load_config
from core.data import check_data, print_split_audit
from core.encoders import MODEL_NAMES
from core.paths import CONFIG
from core.plotting import AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, MODELS, class_name, model_name, save
from core.study import load_study

PLATEAU_TOL = 0.02  # plateau N: smallest N whose mean AUC is within this of the best
CEILING = 0.97      # zero-shot and best AUC both at least this: a tie at the ceiling


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, study, models: list[str], classes: list[str]):
    problems, notes = check_data(study.manifest, cfg.data_path)
    for note in notes:
        print(f"Note: {note}")
    if problems:
        print(f"Data check failed for {cfg.data_path}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    splits = {}
    for c in classes:
        if not study.splits[c].test_pos:
            print(f"Skipping {c}: no recordings could be held out for testing")
            continue
        splits[c] = study.splits[c]
    print_split_audit(study.partition, splits, cfg.training_sizes)

    out = cfg.results_path / "saturation"
    for model in models:
        todo = [c for c in splits if not (out / "runs" / f"{c}_{model}.csv").exists()]
        if not todo:
            print(f"\n[{model}] all classes done (delete results/saturation/runs/*_{model}.csv to rerun)")
            continue
        print(f"\n{'=' * 80}\n[{model}] {len(todo)} classes\n{'=' * 80}")
        start = time.time()
        table, X = study.embed(model, [f for c in todo for f in splits[c].files])
        print(f"[{model}] training heads on {X.device}")
        for c in todo:
            rows, preds = run_class(c, model, X, table, splits[c], cfg.training_sizes, cfg,
                                    cfg.zero_shot_labels.get(c, {}).get(model), low_sample=c in study.partition.low_sample)
            save_csv(preds, out / "predictions" / f"{c}_{model}.csv.gz")
            save_csv(rows, out / "runs" / f"{c}_{model}.csv")  # written last: marks the class x model as done
        del X
        print(f"[{model}] done in {time.time() - start:.0f}s")


# ── Summary: known vs new classes ─────────────────────────────────────────────

def summarise(cfg, study, auc: pd.DataFrame) -> pd.DataFrame:
    """Per class x model: what the curve reaches, where it levels off, and how it compares with a built-in label."""
    partition = study.partition
    rows = []
    for c in cfg.classes:
        train_recs = len({partition.source[f] for f in partition.train_clips[c]})
        for m in [m for m in MODELS if m in cfg.models]:
            d = auc[(auc["class"] == c) & (auc.model == m)].sort_values("training_size")
            if d.empty:
                continue
            trained = d[d.training_size > 0]
            known = bool(cfg.zero_shot_labels.get(c, {}).get(m))
            zs = d[d.training_size == 0].test_auc_mean
            zs = float(zs.iloc[0]) if known and len(zs) else np.nan
            best = trained.loc[trained.test_auc_mean.idxmax()]
            at_n = trained[trained.training_size == cfg.headline_n]
            above = trained[trained.test_auc_mean - trained.test_auc_ci_95 > zs] if known else trained.iloc[:0]
            if not known:
                outcome = ""
            elif zs >= CEILING and best.test_auc_mean >= CEILING:
                outcome = "tie at ceiling"
            elif len(above):
                outcome = "adaptation beats label"
            elif best.test_auc_mean > zs:
                outcome = "adaptation >= label (CI overlaps)"
            else:
                outcome = "label better"
            rows.append({
                "class": c, "model": m, "label_status": "known" if known else "new", "zero_shot_auc": zs,
                f"auc_at_{cfg.headline_n}": float(at_n.test_auc_mean.iloc[0]) if len(at_n) else np.nan,
                f"ci_{cfg.headline_n}": float(at_n.test_auc_ci_95.iloc[0]) if len(at_n) else np.nan,
                "best_auc": float(best.test_auc_mean), "best_n": int(best.training_size),
                "plateau_n": int(trained[trained.test_auc_mean >= best.test_auc_mean - PLATEAU_TOL].training_size.min()),
                "break_even_n": int(above.training_size.min()) if len(above) else np.nan,
                "outcome": outcome, "low_sample": bool(d.low_sample.iloc[0]),
                "train_recordings": train_recs, "train_clips": len(partition.train_clips[c]),
            })
    return pd.DataFrame(rows)


def write_summary(cfg, s: pd.DataFrame):
    out = cfg.results_path / "saturation"
    s.round(4).to_csv(out / "summary.csv", index=False)
    md = s.copy()
    md["class"] = md["class"].map(class_name)
    md["model"] = md["model"].map(model_name)
    md = md.round(3).astype(object).where(md.notna(), "")
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in md.itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Written: {out / 'summary.csv'} (+ .md)")

    known, full = s[s.label_status == "known"], s[~s.low_sample]
    new_everywhere = [c for c in cfg.classes if (s[s["class"] == c].label_status == "new").all()]
    print(f"\nclass x model pairs: {len(s)} | with a built-in label (known): {len(known)} | new: {len(s) - len(known)}")
    print(f"classes no model knows: {new_everywhere}")
    print("known pairs by outcome:", known.outcome.value_counts().to_dict())
    print(f"median break-even N (where adaptation clears the label): {known.break_even_n.median()}")
    print(f"well-sampled classes: median plateau N {full.plateau_n.median()} (within {PLATEAU_TOL} of best)")
    print(f"AUC at N={cfg.headline_n}, well-sampled classes, mean by model:")
    print(full.groupby("model")[f"auc_at_{cfg.headline_n}"].mean()
          .reindex([m for m in MODELS if m in cfg.models]).round(3).to_string())


# ── Figures ───────────────────────────────────────────────────────────────────

CURVE_STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.linewidth": 1.0,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "grid.linewidth": 0.6, "grid.color": GRID, "lines.linewidth": 2, "lines.markersize": 6,
}
BAND_ALPHA = 0.15  # 95% CI bands: up to six overlap per panel, so keep them light
# Training sizes double (5, 10, ..., 160), so x is plotted as log2(N): equal steps per doubling.
# Zero-shot (N = 0) has no log position; it sits one step left of the smallest size, past a separator.
ZS_GAP = 1.2
LOW_SAMPLE_TITLE = "Low-sample classes (provisional: test sets of 3–30 clips from 1–13 recordings)"
# Heatmap: single-hue sequential ramp (validated reference palette, blue steps 100 -> 700): light = low AUC.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
VMIN, VMAX = 0.5, 1.0


def plot_curves(df, classes, path, label_map, suptitle=None):
    """Test AUC against N per class (one panel each), mean and 95% CI band per model, zero-shot as open markers."""
    with plt.rc_context(CURVE_STYLE):
        n_classes = len(classes)
        n_cols = 3 if n_classes > 6 else 2
        n_rows = math.ceil(n_classes / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)
            axes[j - n_cols].xaxis.set_tick_params(labelbottom=True)  # panel above an empty slot keeps its x labels

        handles = {}
        sizes = sorted(s for s in df["training_size"].unique() if s > 0)
        zs_x = log2(sizes[0]) - ZS_GAP
        for ax, cls in zip(axes, classes):
            class_data = df[df["class"] == cls]
            for model in [m for m in MODELS if m in set(class_data["model"])]:
                name, color, marker = MODELS[model]
                model_data = class_data[class_data["model"] == model].sort_values("training_size")
                trained = model_data[model_data["training_size"] > 0]
                x = [log2(n) for n in trained["training_size"]]
                y = trained["test_auc_mean"].to_numpy()
                ci = trained["test_auc_ci_95"].fillna(0).to_numpy()
                # 95% CI as a shaded band (a small box when a class has only one training size)
                band_x = x if len(x) > 1 else [x[0] - 0.2, x[0] + 0.2]
                band_lo, band_hi = (y - ci, y + ci) if len(x) > 1 else ([y[0] - ci[0]] * 2, [y[0] + ci[0]] * 2)
                ax.fill_between(band_x, band_lo, band_hi, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)
                line, = ax.plot(x, y, color=color, marker=marker, markersize=6.5, linewidth=2,
                                markeredgecolor="white", markeredgewidth=1.0, label=name, zorder=3)
                handles.setdefault(model, line)
                zero = model_data[model_data["training_size"] == 0]
                if len(zero):  # zero-shot: open marker, same shape and colour
                    ax.plot(zs_x, zero["test_auc_mean"].iloc[0], marker=marker, markersize=7.5, mfc="white",
                            mec=color, mew=1.8, linestyle="none", zorder=4)

            test_n = class_data["test_size_per_class"].iloc[0]
            title = cls.replace("_", " ").title()
            if test_n != 50:
                title += f" (test n={test_n})"
            knowers = [model_name(m).split(" (")[0] for m in MODELS if label_map.get(cls, {}).get(m)]
            subtitle = ("built-in label: " + ", ".join(knowers)) if knowers else "no model has a built-in label"
            subtitle = textwrap.fill(subtitle, 44)
            ax.set_title(title, fontweight="semibold", pad=12 + 10 * subtitle.count("\n") + 10, color=INK)
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                    color=INK_SECONDARY, linespacing=1.2)
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.set_ylim(0.2, 1.02)
            ax.set_xlim(zs_x - 0.5, log2(sizes[-1]) + 0.4)
            ax.set_xticks([zs_x, *map(log2, sizes)], ["0", *map(str, sizes)])
            ax.axvline(zs_x + ZS_GAP / 2, color=AXIS, linewidth=1, zorder=0)  # separates zero-shot from the curve
            ax.axhline(y=0.5, color=INK_MUTED, linestyle=":", linewidth=1, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.supxlabel("Training size (clips per class, log scale); shaded = 95% CI over 10 random draws; "
                      "open markers = zero-shot", fontweight="medium", fontsize=12, y=0.0)
        fig.supylabel("Test ROC-AUC", fontweight="medium", fontsize=12, x=0.02)
        order = [m for m in MODELS if m in handles]
        fig.legend([handles[m] for m in order], [MODELS[m][0] for m in order], loc="upper center",
                   bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=True, fancybox=False, edgecolor=AXIS,
                   framealpha=0.95, fontsize=10)
        if suptitle:
            fig.suptitle(suptitle, fontweight="semibold", fontsize=13, y=0.95)
        plt.subplots_adjust(hspace=0.45, wspace=0.1, bottom=0.08 if n_rows > 1 else 0.14)
        save(fig, path, pad_inches=0.3, facecolor="white")


def plot_heatmap(cfg, auc: pd.DataFrame, summary: pd.DataFrame):
    """Classes x models at the headline N (low-sample classes at their largest N); built-in labels outlined."""
    n_head = cfg.headline_n
    rows = []
    for c in cfg.classes:
        r = auc[auc["class"] == c]
        for m in r.model.unique():
            t = r[(r.model == m) & (r.training_size > 0)]
            n = n_head if n_head in set(t.training_size) else int(t.training_size.max())
            rows.append({"class": c, "model": m, "n": n, "auc": float(t[t.training_size == n].test_auc_mean.iloc[0])})
    cells = pd.DataFrame(rows).merge(summary[["class", "model", "label_status", "zero_shot_auc", "low_sample"]],
                                     on=["class", "model"])
    models = [m for m in MODELS if m in set(cells.model)]
    low = set(cells[cells.low_sample]["class"])
    classes = [c for c in cfg.classes if c in set(cells["class"]) and c not in low] + \
              [c for c in cfg.classes if c in low]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", RAMP)

    fig, ax = plt.subplots(figsize=(1.35 * len(models) + 2.6, 0.62 * len(classes) + 1.6))
    grid = np.full((len(classes), len(models)), np.nan)
    for _, row in cells.iterrows():
        grid[classes.index(row["class"]), models.index(row.model)] = row.auc
    im = ax.imshow(grid, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    for _, row in cells.iterrows():
        i, j = classes.index(row["class"]), models.index(row.model)
        ink = "white" if (row.auc - VMIN) / (VMAX - VMIN) > 0.55 else INK
        label = f"{row.auc:.2f}" + (f" @{row.n}" if row.n != n_head else "")
        known = row.label_status == "known"
        ax.text(j, i - (0.12 if known else 0), label, ha="center", va="center", fontsize=8.5, color=ink,
                fontweight="semibold")
        if known:
            ax.text(j, i + 0.24, f"zs {row.zero_shot_auc:.2f}", ha="center", va="center", fontsize=7, color=ink)
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, linewidth=1.8))

    if low:
        ax.axhline(len(classes) - len(low) - 0.5, color=INK, linewidth=1.5)
    ax.set_xticks(range(len(models)), [model_name(m).replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_yticks(range(len(classes)), [class_name(c) + (" *" if c in low else "") for c in classes], fontsize=9)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"Test ROC-AUC at N = {n_head} clips per class", color=INK)
    cbar.outline.set_visible(False)
    fig.text(0.01, 0.01, "Outlined = model has a built-in label for the class (zs = its zero-shot AUC); "
             "* low sample (provisional), @N = largest N reached.", fontsize=8, color=INK_SECONDARY)
    save(fig, cfg.figs_path / "saturation_summary", facecolor="white")


def plot(cfg):
    out = cfg.results_path / "saturation"
    auc = pd.read_csv(out / "auc_by_n.csv")
    present = [c for c in cfg.classes if c in set(auc["class"])]
    low = set(auc.loc[auc.low_sample.astype(bool), "class"])
    groups = {"main": [c for c in present if c not in low], "low_sample": [c for c in present if c in low]}
    for group, classes in groups.items():
        if classes:
            plot_curves(auc, classes, cfg.figs_path / f"saturation_{group}", cfg.zero_shot_labels,
                        LOW_SAMPLE_TITLE if group == "low_sample" else None)
    plot_heatmap(cfg, auc, pd.read_csv(out / "summary.csv"))


def main():
    parser = argparse.ArgumentParser(description="Training-size (saturation) experiment.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="Run only these models (default: all in "
                        "the config). Lets CPU (TensorFlow) and GPU (PyTorch) models run as two parallel jobs.")
    parser.add_argument("--classes", nargs="+", help="Run only these classes (the split is still built from every "
                        "class in the config, so test sets are unchanged).")
    parser.add_argument("--plot", action="store_true", help="Only redraw the figures from saved results.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep nohup/redirected logs complete
    cfg = load_config(args.config)
    if not args.plot:
        if args.classes and set(args.classes) - set(cfg.classes):
            parser.error(f"--classes must be in the config's classes list: {sorted(set(args.classes) - set(cfg.classes))}")
        study = load_study(cfg)
        train(cfg, study, args.models or cfg.models, args.classes or cfg.classes)
        print("\nAggregating results...")
        runs_dir = cfg.results_path / "saturation" / "runs"
        auc = pd.concat([aggregate(r) for c in cfg.classes if (r := load_runs(runs_dir, c)) is not None],
                        ignore_index=True)
        auc.to_csv(cfg.results_path / "saturation" / "auc_by_n.csv", index=False)
        write_summary(cfg, summarise(cfg, study, auc))
    plot(cfg)


if __name__ == "__main__":
    main()
