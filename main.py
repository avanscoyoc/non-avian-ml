"""Training-size experiment: frozen embeddings + MLP heads on a test set held out by source recording.

One train/test partition by recording is shared with the multiclass experiment. For each model, every
clip's embedding is computed once and cached; then for each class, training size N and seed, a head is
trained on N positive + N negative clips drawn from the training recordings and scored on the class's
fixed test clips (all seeds for a size are trained at once). The 10 seeds give the mean and 95% CI.
Zero-shot (N=0) uses the model's own label scores.

Usage:
    pixi run python main.py                       # laptop (CPU)
    pixi run -e gpu python main.py                # cirrus (PyTorch models + heads on GPU)
    pixi run python main.py --config my.yaml      # another config

    # cirrus: TensorFlow models (CPU) and PyTorch models (GPU) as two parallel jobs
    pixi run -e gpu python main.py --models birdnet perch perch2
    pixi run -e gpu python main.py --models birdnet3 cnn14 resnet
"""
import argparse
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.paths import FIGS, ROOT
from src.config import build_partition, load_config
from src.data_loader import binary_split, check_data, load_audio_files, load_manifest, print_split_audit
from src.embeddings import get_embeddings
from src.model_loader import MODEL_NAMES, torch_device
from src.trainer import predict_heads, train_heads
from src.zero_shot import load_label_map, match_labels, zero_shot_scores
from src.results import (
    aggregate_results, plot_species_models, save_classifier_bundle, save_predictions, save_results, save_runs,
)


def run_class(species, model_name, X, table, split, sizes, cfg, label_map, low_sample=False):
    """All training sizes x seeds (+ zero-shot) for one class and model -> (run rows, prediction rows)."""
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
        draws = [load_audio_files(size, split.train_pos, split.train_neg, seed) for seed in cfg.random_seeds]
        heads = train_heads(X, [table.rows(f) for f, _ in draws], [l for _, l in draws],
                            batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, seed=size)
        probs = predict_heads(heads, X, [test_rows] * len(draws))
        aucs = [record(size, seed, p[:, 1]) for seed, p in zip(cfg.random_seeds, probs)]
        for k, seed in enumerate(cfg.random_seeds):
            rows.append({**base, "training_size": size, "random_seed": seed, "test_auc": aucs[k]})
            if cfg.save_classifier:
                save_classifier_bundle(
                    classifier=heads.head(k), embedding_model=None, model_name=model_name, species=species,
                    training_size=size, random_seed=seed, test_auc=aucs[k], n_epochs=cfg.n_epochs,
                    results_path=cfg.results_path,
                )
        print(f"  {species:20s} N={size:<4d} test AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    wanted = label_map.get(species, {}).get(model_name)
    if cfg.run_zeroshot and wanted and table.labels:
        scores = zero_shot_scores(np.asarray(table.scores)[test_rows], match_labels(table.labels, wanted))
        auc = record(0, 0, scores)
        rows.append({**base, "training_size": 0, "n_epochs": 0, "random_seed": 0, "test_auc": auc})
        print(f"  {species:20s} zero-shot ({', '.join(wanted)}): test AUC {auc:.3f}")

    pred_rows = [{"species": species, "model": model_name, "training_size": p["training_size"],
                  "random_seed": p["random_seed"], "filename": f, "label": y, "score": float(s)}
                 for p in preds for f, y, s in zip(test_files, test_labels, p["scores"])]
    return rows, pred_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "src" / "config.yaml"))
    parser.add_argument("--models", nargs="+", help="Run only these models (default: all in the config). "
                        "Lets CPU (TensorFlow) and GPU (PyTorch) models run as two parallel jobs.")
    parser.add_argument("--species", nargs="+", help="Run only these classes (the partition is still built "
                        "from every class in the config, so test sets are unchanged).")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep nohup/redirected logs complete
    cfg = load_config(args.config)
    if args.models:
        unknown = set(args.models) - set(MODEL_NAMES)
        if unknown:
            parser.error(f"unknown models {sorted(unknown)}; choose from {MODEL_NAMES}")
        cfg.models = args.models

    manifest = load_manifest(cfg.clips_csv)
    problems, notes = check_data(manifest, cfg.data_path)
    for note in notes:
        print(f"Note: {note}")
    if problems:
        print(f"Data check failed for {cfg.data_path}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    partition = build_partition(manifest, cfg)
    if args.species and set(args.species) - set(cfg.species):
        parser.error(f"--species must be in the config's species list: {sorted(set(args.species) - set(cfg.species))}")
    splits, sizes = {}, {}
    for sp in args.species or cfg.species:
        split = binary_split(partition, sp, cfg.split_seed)
        if not split.test_pos:
            print(f"Skipping {sp}: no recordings could be held out for testing")
            continue
        splits[sp] = split
        if cfg.species_sizes and sp in cfg.species_sizes:
            sizes[sp] = [min(cfg.species_sizes[sp]["train"], split.max_n)]
        else:
            sizes[sp] = cfg.training_sizes
    print_split_audit(partition, splits, sorted({n for s in sizes.values() for n in s}))

    label_map = load_label_map() if cfg.run_zeroshot else {}
    runs_dir, preds_dir = cfg.results_path / "runs", cfg.results_path / "predictions"
    device = torch_device()
    cls_of = dict(zip(manifest.filename, manifest.cls))

    for model_name in cfg.models:
        todo = [sp for sp in splits if not (runs_dir / f"{sp}_{model_name}.csv").exists()]
        if not todo:
            print(f"\n[{model_name}] all classes done (delete results/runs/*_{model_name}.csv to rerun)")
            continue
        print(f"\n{'=' * 80}\n[{model_name}] {len(todo)} classes\n{'=' * 80}")
        start = time.time()
        clips = {(cls_of[f], f) for sp in todo for s in [splits[sp]]
                 for f in s.train_pos + s.train_neg + s.test_pos + s.test_neg}
        table = get_embeddings(model_name, sorted(clips), cfg.data_path, cfg.embeddings_path)
        X = torch.from_numpy(np.ascontiguousarray(table.emb)).to(device)
        print(f"[{model_name}] training heads on {device}")
        for sp in todo:
            rows, pred_rows = run_class(sp, model_name, X, table, splits[sp], sizes[sp], cfg, label_map,
                                        low_sample=sp in partition.low_sample)
            save_predictions(pred_rows, preds_dir, sp, model_name)
            save_runs(rows, runs_dir, sp, model_name)  # written last: marks the class x model as done
        del X
        print(f"[{model_name}] done in {time.time() - start:.0f}s")

    print("\nAggregating results...")
    FIGS.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        all_runs = aggregate_results(runs_dir, sp)
        if all_runs is None:
            continue
        df = save_results(all_runs.to_dict("records"), cfg.results_path / f"results_{sp}.csv")
        plot_species_models(df, FIGS / f"results_{sp}_plot.png")
    print(f"Results: {cfg.results_path}/results_*.csv | per-class plots: {FIGS}/results_*_plot.png")


if __name__ == "__main__":
    main()
