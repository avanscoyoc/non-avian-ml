"""Binary detector learning curves, shared by the saturation and window-length experiments.

For one target class and model: heads at every training size x random draw, scored on the class's fixed test
clips, plus the zero-shot point from the model's own label for the class (where it has one). Each draw's AUC is a
run row; `aggregate` turns run rows into mean, SD and 95% CI per class x model x N.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.heads import predict_heads, train_heads
from core.zero_shot import match_labels, zero_shot_scores

GROUP = ["model", "species", "training_size", "n_epochs", "batch_size", "test_size_per_class", "low_sample"]


def run_class(species, model_name, X, table, split, sizes, cfg, zero_shot_labels=None, low_sample=False):
    """All training sizes x seeds (+ zero-shot, if given the model's labels for the class) -> (run rows, prediction rows)."""
    test_files = split.test_pos + split.test_neg
    test_labels = [1] * len(split.test_pos) + [0] * len(split.test_neg)
    test_rows = table.rows(test_files)
    base = {"species": species, "model": model_name, "n_epochs": cfg.n_epochs, "batch_size": cfg.batch_size,
            "test_size_per_class": len(split.test_pos), "low_sample": low_sample}
    rows, preds = [], []

    def record(size, seed, scores):
        preds.append({"training_size": size, "random_seed": seed, "scores": scores})
        return roc_auc_score(test_labels, scores)

    skipped = [n for n in sizes if n > split.max_n]
    if skipped:
        print(f"  {species:20s} N={', '.join(map(str, skipped))}: skipped (training pool {split.max_n} clips per side)")
    for size in sizes:
        if size > split.max_n:
            continue
        draws = split.draws(size, cfg.random_seeds)
        heads = train_heads(X, [table.rows(f) for f, _ in draws], [l for _, l in draws],
                            batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, seed=size)
        probs = predict_heads(heads, X, test_rows)[:, :, 1]
        aucs = [record(size, seed, p) for seed, p in zip(cfg.random_seeds, probs)]
        rows += [{**base, "training_size": size, "random_seed": seed, "test_auc": auc}
                 for seed, auc in zip(cfg.random_seeds, aucs)]
        print(f"  {species:20s} N={size:<4d} test AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    if zero_shot_labels and table.labels:
        scores = zero_shot_scores(np.asarray(table.scores)[test_rows], match_labels(table.labels, zero_shot_labels))
        auc = record(0, 0, scores)
        rows.append({**base, "training_size": 0, "n_epochs": 0, "random_seed": 0, "test_auc": auc})
        print(f"  {species:20s} zero-shot ({', '.join(zero_shot_labels)}): test AUC {auc:.3f}")

    pred_rows = [{"species": species, "model": model_name, "training_size": p["training_size"],
                  "random_seed": p["random_seed"], "filename": f, "label": y, "score": float(s)}
                 for p in preds for f, y, s in zip(test_files, test_labels, p["scores"])]
    return rows, pred_rows


def save_csv(rows, path) -> Path:
    """Rows to CSV (gzipped if the name ends in .gz), creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gzip = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None  # no timestamp: same rows, same bytes
    pd.DataFrame(rows).to_csv(path, index=False, compression=gzip)
    return path


def load_runs(runs_dir, species) -> pd.DataFrame | None:
    """Every model's run rows for one class."""
    files = sorted(Path(runs_dir).glob(f"{species}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Test AUC over the random training draws per class x model x N: mean, SD, 95% CI (1.96 SD / sqrt(n)), n."""
    if "low_sample" not in runs:
        runs = runs.assign(low_sample=False)
    agg = runs.groupby(GROUP)["test_auc"].agg(["mean", "std", "count"]).round(4)
    agg.columns = ["test_auc_mean", "test_auc_std", "test_auc_n"]
    agg = agg.reset_index()
    agg["test_auc_ci_95"] = (1.96 * agg["test_auc_std"] / np.sqrt(agg["test_auc_n"])).round(4)
    return agg[GROUP + ["test_auc_mean", "test_auc_std", "test_auc_ci_95", "test_auc_n"]]
