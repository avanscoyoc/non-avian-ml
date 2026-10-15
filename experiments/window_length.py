"""Input window length, Figure 7 and Table S3 of docs/brief.md: does a longer BirdNET 3.0 input window help long,
broadband noise more than short, pulsed animal sounds?

Data: data/windows/{3s,5s,10s}/{class}/{clip_id}.wav (the same clips, centred on the same event, cut at three
lengths by scripts/make_window_clips.py) and data/windows/manifest.csv (clip -> source recording).

Design (the same method as the saturation experiment, so only the window length changes):
- Classes: broadband (device static, generator, wind, traffic) vs pulsed (bullfrog, field cricket, Pacific chorus
  frog). Negatives come from every other class in the window set (equal quota per class).
- One split by source recording, shared by all three window lengths: 25 test clips per class (+ 25 negatives),
  training clips only from training recordings, so the 3/5/10 s comparison is paired clip-for-clip.
- N = 5, 10, 20 clips per class, 10 random draws, the same MLP head.
- Only clips that are used (test clips + every training draw) are encoded, once per window length.

Outputs: results/window_length/{runs,predictions}/{3s,5s,10s}/, results/window_length/auc_by_n.csv.
Figure: window_size.

Usage:
    pixi run window-length                                  # encode + train what's missing, then plot
    pixi run python -m experiments.window_length --plot     # figure only
"""
import argparse
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.binary import aggregate, load_runs, run_class, save_csv
from core.config import load_config
from core.data import print_split_audit
from core.paths import CONFIG, DATA
from core.plotting import INK, INK_MUTED, class_name, save, style_axes
from core.study import make_study

WINDOW_DATA = DATA / "windows"
MODEL = "birdnet3"
WINDOWS = {"3s": 3.0, "5s": 5.0, "10s": 10.0}
GROUPS = {"broadband": ["device_static", "generator", "wind", "traffic"],
          "pulsed": ["american_bullfrog", "field_cricket", "pacific_chorus_frog"]}
GROUP_OF = {c: g for g, classes in GROUPS.items() for c in classes}
CLASSES = list(GROUP_OF)
SIZES = [5, 10, 20]
TEST_SIZE, MIN_TRAIN = 25, 20


def load_window_study(cfg):
    """The window clip set (data/windows/manifest.csv) and its own split by source recording."""
    m = pd.read_csv(WINDOW_DATA / "manifest.csv")
    manifest = pd.DataFrame({"cls": m.common_name, "filename": m.clip_id + ".wav", "source": m.source_audio_path})
    return make_study(manifest, CLASSES, cfg, TEST_SIZE, MIN_TRAIN)


def run(cfg, out):
    study = load_window_study(cfg)
    print_split_audit(study.partition, study.splits, SIZES)

    # Only the clips that are used: each class's test set and every training draw.
    needed = set()
    for s in study.splits.values():
        needed |= set(s.test_files)
        for n in SIZES:
            if n <= s.max_n:
                needed |= {f for files, _ in s.draws(n, cfg.random_seeds) for f in files}
    print(f"{len(needed)} clips needed per window size")

    for w, seconds in WINDOWS.items():
        runs_dir = out / "runs" / w
        todo = [c for c in CLASSES if not (runs_dir / f"{c}_{MODEL}.csv").exists()]
        if not todo:
            print(f"[{w}] done")
            continue
        print(f"\n[{w}] {seconds:g} s windows")
        window = replace(study, data_path=WINDOW_DATA / w, embeddings_path=cfg.embeddings_path / "windows" / w,
                         clip_seconds=seconds)
        table, X = window.embed(MODEL, needed)
        for c in todo:
            rows, preds = run_class(c, MODEL, X, table, study.splits[c], SIZES, cfg,
                                    low_sample=c in study.partition.low_sample)
            save_csv(preds, out / "predictions" / w / f"{c}_{MODEL}.csv.gz")
            save_csv(rows, runs_dir / f"{c}_{MODEL}.csv")

    frames = [aggregate(runs).assign(window=w, group=GROUP_OF[c])
              for w in WINDOWS for c in CLASSES if (runs := load_runs(out / "runs" / w, c)) is not None]
    pd.concat(frames).to_csv(out / "auc_by_n.csv", index=False)
    print(f"\nResults: {out / 'auc_by_n.csv'}")


def plot(cfg, out):
    df = pd.read_csv(out / "auc_by_n.csv")
    secs = [WINDOWS[w] for w in WINDOWS]
    n_colors = {5: "#9ec5f4", 10: "#3987e5", 20: "#104281"}  # one-hue ramp: lighter = fewer clips
    n_cols = max(len(c) for c in GROUPS.values()) + 1  # classes, then a group-mean panel at the end of each row
    fig, axes = plt.subplots(len(GROUPS), n_cols, figsize=(3.1 * n_cols, 3.4 * len(GROUPS)), sharey=True,
                             constrained_layout=True)
    for r, (group, classes) in enumerate(GROUPS.items()):
        panels = [(class_name(c), df[df["class"] == c]) for c in classes]
        mean = df[df.group == group].groupby(["window", "training_size"], as_index=False).test_auc_mean.mean()
        panels.append((f"{group.capitalize()} mean ({len(classes)} classes)", mean.assign(test_auc_ci_95=np.nan)))
        for k, ax in enumerate(axes[r]):
            if k == len(panels) - 1 or k >= len(classes):  # mean goes in the last column; hide any gap before it
                if k < n_cols - 1:
                    ax.set_visible(False)
                    continue
                title, d = panels[-1]
            else:
                title, d = panels[k]
            for n in SIZES:
                dn = d[d.training_size == n].set_index("window").reindex(list(WINDOWS))
                y = dn.test_auc_mean.to_numpy()
                ax.plot(secs, y, marker="o", color=n_colors[n], linewidth=2, markeredgecolor="white", label=f"N = {n}")
                ci = dn.test_auc_ci_95.to_numpy()
                if not np.all(np.isnan(ci)):
                    ax.fill_between(secs, y - ci, y + ci, color=n_colors[n], alpha=0.15, linewidth=0)
            ax.set_title(title, color=INK, fontweight="semibold", fontsize=10)
            ax.set_xticks(secs, [f"{s:g} s" for s in secs])
            style_axes(ax)
            ax.axhline(0.5, color=INK_MUTED, linestyle=":", linewidth=1)
        axes[r][0].set_ylabel(f"{group.capitalize()}\nTest ROC-AUC", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("BirdNET 3.0 input window: test AUC by window length (shaded = 95% CI over 10 draws)",
                 color=INK, fontweight="semibold")
    save(fig, cfg.figs_path / "window_size", facecolor="white")


def main():
    parser = argparse.ArgumentParser(description="BirdNET 3.0 input window length.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--plot", action="store_true", help="Only redraw the figure from saved results.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(args.config)
    out = cfg.results_path / "window_length"
    if not args.plot:
        run(cfg, out)
    plot(cfg, out)


if __name__ == "__main__":
    main()
