"""
window_size.py

BirdNET 3.0 with 3, 5 and 10 s input windows: does a longer window help long, broadband noise more than
short, pulsed animal sounds?

Data: birdnet3_windows/{3s,5s,10s}/{class}/{clip_id}.wav (the same clips, centred on the same event, cut at
three lengths) and birdnet3_windows/manifest.csv (clip -> source recording). See birdnet3_windows/README.md.

Design (same method as the training-size experiment, so only the window length changes):
- Classes: broadband (device static, generator, wind, traffic) vs pulsed (bullfrog, field cricket, Pacific
  chorus frog). Negatives come from every other class in the window set (equal quota per class).
- One split by source recording, shared by all three window sizes: 25 test clips per class (+ 25 negatives),
  training clips only from training recordings, so the 3/5/10 s comparison is paired clip-for-clip.
- N = 5, 10, 20 clips per class, 10 random draws, the same MLP head as main.py.
- Only clips that are actually used (test clips + every training draw) are encoded, once per window size.

Outputs: results/windows/{runs,predictions}/, results/windows/window_results.csv, figs/window_size.png/.pdf

Usage:
    pixi run python experiment/window_size.py            # encodes + trains, then plots
    pixi run python experiment/window_size.py --plot     # plot only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import FIGS, RESULTS, ROOT  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_loader import binary_split, load_audio_files, make_partition, print_split_audit  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.model_loader import torch_device  # noqa: E402
from src.results import aggregate_results, save_predictions, save_results, save_runs  # noqa: E402
import main  # noqa: E402  (run_class: same training/evaluation as the training-size experiment)

WINDOW_DATA = ROOT / "birdnet3_windows"
OUT = RESULTS / "windows"
MODEL = "birdnet3"
WINDOWS = {"3s": 3.0, "5s": 5.0, "10s": 10.0}
GROUPS = {"broadband": ["device_static", "generator", "wind", "traffic"],
          "pulsed": ["american_bullfrog", "field_cricket", "pacific_chorus_frog"]}
CLASSES = [c for g in GROUPS.values() for c in g]
SIZES = [5, 10, 20]
TEST_SIZE, MIN_TRAIN = 25, 20


def window_manifest() -> pd.DataFrame:
    m = pd.read_csv(WINDOW_DATA / "manifest.csv")
    return pd.DataFrame({"cls": m.common_name, "filename": m.clip_id + ".wav", "source": m.source_audio_path})


def run(cfg):
    manifest = window_manifest()
    partition = make_partition(manifest, CLASSES, TEST_SIZE, cfg.split_seed, cfg.max_test_recording_clips,
                               {c: MIN_TRAIN for c in CLASSES}, cfg.donor_test_fraction, cfg.low_sample_test_fraction)
    splits = {c: binary_split(partition, c, cfg.split_seed) for c in CLASSES}
    print_split_audit(partition, splits, SIZES)

    # Only the clips that are used: each class's test set and every training draw.
    needed = set()
    for s in splits.values():
        needed |= set(s.test_pos + s.test_neg)
        for n in SIZES:
            if n <= s.max_n:
                for seed in cfg.random_seeds:
                    needed |= set(load_audio_files(n, s.train_pos, s.train_neg, seed)[0])
    cls_of = dict(zip(manifest.filename, manifest.cls))
    clips = sorted((cls_of[f], f) for f in needed)
    print(f"{len(clips)} clips needed per window size")

    cfg.run_zeroshot = False
    device = torch_device()
    for w, seconds in WINDOWS.items():
        runs_dir, preds_dir = OUT / "runs" / w, OUT / "predictions" / w
        todo = [c for c in CLASSES if not (runs_dir / f"{c}_{MODEL}.csv").exists()]
        if not todo:
            print(f"[{w}] done")
            continue
        print(f"\n[{w}] {seconds:g} s windows")
        table = get_embeddings(MODEL, clips, WINDOW_DATA / w, OUT / "embeddings" / w, clip_seconds=seconds)
        X = torch.from_numpy(np.ascontiguousarray(table.emb)).to(device)
        for c in todo:
            rows, preds = main.run_class(c, MODEL, X, table, splits[c], SIZES, cfg, {},
                                         low_sample=c in partition.low_sample)
            save_predictions(preds, preds_dir, c, MODEL)
            save_runs(rows, runs_dir, c, MODEL)

    frames = []
    for w in WINDOWS:
        for c in CLASSES:
            runs = aggregate_results(OUT / "runs" / w, c)
            if runs is not None:
                agg = save_results(runs.to_dict("records"), OUT / "tmp_results.csv")
                frames.append(agg.assign(window=w, group=next(g for g, cl in GROUPS.items() if c in cl)))
    (OUT / "tmp_results.csv").unlink(missing_ok=True)
    pd.concat(frames).to_csv(OUT / "window_results.csv", index=False)
    print(f"\nResults: {OUT / 'window_results.csv'}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plot_style import AXIS, GRID, INK, INK_MUTED, INK_SECONDARY, class_name

    df = pd.read_csv(OUT / "window_results.csv")
    secs = [WINDOWS[w] for w in WINDOWS]
    n_colors = {5: "#9ec5f4", 10: "#3987e5", 20: "#104281"}  # one-hue ramp: lighter = fewer clips
    n_cols = max(len(c) for c in GROUPS.values()) + 1  # classes, then a group-mean panel at the end of each row
    fig, axes = plt.subplots(len(GROUPS), n_cols, figsize=(3.1 * n_cols, 3.4 * len(GROUPS)), sharey=True,
                             constrained_layout=True)
    for r, (group, classes) in enumerate(GROUPS.items()):
        panels = [(class_name(c), df[df.species == c]) for c in classes]
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
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.axhline(0.5, color=INK_MUTED, linestyle=":", linewidth=1)
            ax.tick_params(colors=INK_SECONDARY)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
        axes[r][0].set_ylabel(f"{group.capitalize()}\nTest ROC-AUC", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("BirdNET 3.0 input window: test AUC by window length (shaded = 95% CI over 10 draws)",
                 color=INK, fontweight="semibold")
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"window_size.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGS / 'window_size'}.{{png,pdf}}")


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Only redraw the figure from saved results.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if not args.plot:
        run(load_config(ROOT / "src" / "config.yaml"))
    plot()


if __name__ == "__main__":
    main_cli()
