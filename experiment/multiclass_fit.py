"""
multiclass_fit.py

Multiclass MLP heads ("which noise is this?") on the same shared train/test partition as the
training-size (binary) experiment. Groups and N come from src/config.yaml (multiclass_groups,
multiclass_train_size = 40); classes with fewer than N training clips are listed as waiting for data
and join automatically once data/metadata.csv has enough.

Test set: each class's held-out clips from the shared partition, the same clips as its binary test
positives. Training, per seed (1-10): exactly the N clips per class that the binary N-clip draw used for
that class and seed. Embeddings come from the shared cache (results/embeddings); all seeds are trained
at once.

Outputs in {results_path}/experiment/ (one set per group x model):
  multiclass_{group}_{model}.csv          per seed x class: test n, one-vs-rest AUC, precision, recall, F1, macro F1
  cm_{group}_{model}.npy                  confusion matrices, shape (seeds, n_classes, n_classes), counts
  test_embeddings_{group}_{model}.npz     test-clip embeddings + labels (for the t-SNE figure)
  labels_{group}.json                     class names (index order) and test clips per class
  multiclass_results.csv                  mean/SD over seeds per group x model x class

Classes with no test clips get NaN metrics and an empty confusion-matrix row, instead of an error.

Usage:
    pixi run python experiment/multiclass_fit.py
    pixi run python experiment/multiclass_fit.py --models birdnet perch2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import binary_split, load_audio_files, load_manifest  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.model_loader import MODEL_NAMES, torch_device  # noqa: E402
from src.trainer import predict_heads, train_heads  # noqa: E402


def training_clips(split, n: int, seed: int) -> list[str]:
    """The n positive clips the binary training-size run drew for this class and seed."""
    files, labels = load_audio_files(n, split.train_pos, split.train_neg, seed)
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


def run_group_model(group, classes, model_name, partition, splits, cfg, cls_of, device, out):
    test_files = [f for c in classes for f in partition.test_clips[c]]
    y_test = np.array([i for i, c in enumerate(classes) for _ in partition.test_clips[c]])
    n = cfg.multiclass_train_size
    draws = []
    for seed in cfg.random_seeds:
        files = [f for c in classes for f in training_clips(splits[c], n, seed)]
        labels = [i for i, c in enumerate(classes) for _ in range(n)]
        draws.append((files, labels))

    clips = {(cls_of[f], f) for f in test_files} | {(cls_of[f], f) for files, _ in draws for f in files}
    table = get_embeddings(model_name, sorted(clips), cfg.data_path, cfg.embeddings_path)
    X = torch.from_numpy(np.ascontiguousarray(table.emb)).to(device)
    test_rows = table.rows(test_files)

    heads = train_heads(X, [table.rows(f) for f, _ in draws], [l for _, l in draws], n_classes=len(classes),
                        batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, seed=n)
    rows, cms = [], []
    for seed, probs in zip(cfg.random_seeds, predict_heads(heads, X, [test_rows] * len(draws))):
        m = evaluate(probs, y_test, len(classes))
        cms.append(m["cm"])
        rows += [{"group": group, "model": model_name, "seed": seed, "class_name": c, "class_idx": i,
                  "test_n": int((y_test == i).sum()), "ovr_auc": m["auc"][i], "precision": m["precision"][i],
                  "recall": m["recall"][i], "f1": m["f1"][i], "macro_f1": m["macro_f1"]}
                 for i, c in enumerate(classes)]

    df = pd.DataFrame(rows)
    df.to_csv(out / f"multiclass_{group}_{model_name}.csv", index=False)
    np.save(out / f"cm_{group}_{model_name}.npy", np.stack(cms))
    np.savez(out / f"test_embeddings_{group}_{model_name}.npz", embeddings=table.emb[test_rows],
             labels=y_test, class_names=np.array(classes))
    print(f"  {group:9s} {model_name:8s} macro F1 {df.groupby('seed').macro_f1.first().mean():.3f} | "
          f"OvR AUC " + ", ".join(f"{c} {df[df.class_name == c].ovr_auc.mean():.3f}" for c in classes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "src" / "config.yaml"))
    parser.add_argument("--group", help="Run one group (default: all in the config).")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="Default: all models in the config.")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cfg = load_config(args.config)
    groups = {args.group: cfg.multiclass_groups[args.group]} if args.group else cfg.multiclass_groups
    models = args.models or cfg.models
    manifest = load_manifest(cfg.clips_csv)
    partition = build_partition(manifest, cfg)
    cls_of = dict(zip(manifest.filename, manifest.cls))
    device = torch_device()
    out = cfg.results_path / "experiment"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Groups: {list(groups)} | models: {models} | seeds: {cfg.random_seeds} | "
          f"N = {cfg.multiclass_train_size} per class | heads on {device}")

    for group, classes in groups.items():
        splits = {c: binary_split(partition, c, cfg.split_seed) for c in classes}
        waiting = {c: len(s.train_pos) for c, s in splits.items() if len(s.train_pos) < cfg.multiclass_train_size}
        if waiting:
            print(f"\n{group}: waiting for more data (fewer than {cfg.multiclass_train_size} training clips): {waiting}")
        classes = [c for c in classes if c not in waiting]
        if len(classes) < 2:
            print(f"{group}: fewer than 2 classes with enough data, skipped")
            continue
        test_n = [len(partition.test_clips[c]) for c in classes]
        (out / f"labels_{group}.json").write_text(json.dumps({"classes": classes, "test_n": test_n}))
        print(f"\n{group}: test clips per class {dict(zip(classes, test_n))}")
        for model_name in models:
            if (out / f"multiclass_{group}_{model_name}.csv").exists():
                print(f"  {group} {model_name}: done (delete multiclass_{group}_{model_name}.csv to rerun)")
                continue
            run_group_model(group, classes, model_name, partition, splits, cfg, cls_of, device, out)

    dfs = [pd.read_csv(p) for p in sorted(out.glob("multiclass_*_*.csv")) if p.name != "multiclass_results.csv"]
    if dfs:
        agg = (pd.concat(dfs).groupby(["group", "model", "class_name"])[["ovr_auc", "f1", "macro_f1"]]
               .agg(["mean", "std"]))
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(out / "multiclass_results.csv", index=False)
        print(f"\nAggregated results -> {out / 'multiclass_results.csv'}")


if __name__ == "__main__":
    main()
