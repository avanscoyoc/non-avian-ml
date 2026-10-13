import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import json
import shutil
from datetime import datetime

from src.paths import MODELS


def save_runs(rows, runs_dir, species, model):
    """Save all runs (every training size x seed, plus zero-shot) for one class x model."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_file = runs_dir / f"{species}_{model}.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return output_file


def aggregate_results(runs_dir, species):
    """Load every model's run CSV for a species."""
    run_files = sorted(Path(runs_dir).glob(f"{species}_*.csv"))
    if not run_files:
        return None
    return pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)


def save_predictions(rows, preds_dir, species, model):
    """Every test clip's score for every training size x seed (and zero-shot), for later bootstraps."""
    preds_dir = Path(preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    output_file = preds_dir / f"{species}_{model}.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    return output_file


def save_results(results, output_path):
    """Aggregate test AUC across the random training draws (seeds) and save to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    if "low_sample" not in df:
        df["low_sample"] = False
    agg_results = (
        df.groupby(["model", "species", "training_size", "n_epochs", "batch_size", "test_size_per_class", "low_sample"])
        ["test_auc"].agg(["mean", "std", "count"]).round(4)
    )
    agg_results.columns = ["test_auc_mean", "test_auc_std", "test_auc_n"]
    agg_results = agg_results.reset_index()

    # 95% CI across seeds: 1.96 * std / sqrt(n)
    agg_results["test_auc_ci_95"] = (
        1.96 * agg_results["test_auc_std"] / np.sqrt(agg_results["test_auc_n"])
    ).round(4)

    column_order = [
        "model", "species", "training_size", "n_epochs", "batch_size", "test_size_per_class", "low_sample",
        "test_auc_mean", "test_auc_std", "test_auc_ci_95", "test_auc_n",
    ]
    agg_results = agg_results[column_order]
    agg_results.to_csv(output_path, index=False)
    return agg_results


def plot_species_models(df, output_path):
    """Plot learning curves with 95% confidence intervals; zero-shot (N=0) as open markers."""
    species_list = df["species"].unique()
    n_species = len(species_list)
    fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 6))
    if n_species == 1:
        axes = [axes]

    for i, species in enumerate(species_list):
        species_data = df[df["species"] == species]
        models = species_data["model"].unique()

        for model in models:
            model_data = species_data[species_data["model"] == model].sort_values("training_size")
            trained = model_data[model_data["training_size"] > 0]
            line = axes[i].errorbar(
                trained["training_size"],
                trained["test_auc_mean"],
                yerr=trained["test_auc_ci_95"],
                label=model.upper(),
                marker="o",
                capsize=5,
                capthick=2,
            )
            zero = model_data[model_data["training_size"] == 0]
            if len(zero):
                axes[i].plot(0, zero["test_auc_mean"].iloc[0], marker="o", mfc="none",
                             color=line[0].get_color(), linestyle="none")

        axes[i].set_title(f"{species.replace('_', ' ').title()}")
        axes[i].set_xlabel("Training Size (samples per class)")
        axes[i].set_ylabel("Test ROC-AUC")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# (model_info, preprocessing) per encoder: the layer the classifier head expects as input.
ENCODER_INFO = {
    "birdnet": ({"type": "birdnet", "version": "2.4", "embedding_dim": 1024,
                 "embedding_layer": "model/GLOBAL_AVG_POOL/Mean"},
                {"sample_rate": 48000, "duration_s": 3.0}),
    "birdnet3": ({"type": "birdnet", "version": "3.0-preview3.1", "embedding_dim": 1280,
                  "source": "birdnet python package, pt backend"},
                 {"sample_rate": 32000, "duration_s": 3.0}),
    "perch": ({"type": "perch", "version": "v8", "embedding_dim": 1280, "embedding_layer": "embedding"},
              {"sample_rate": 32000, "duration_s": 5.0, "padding": "3 s clip centre-padded to 5 s"}),
    "perch2": ({"type": "perch", "version": "2.0", "embedding_dim": 1536, "embedding_layer": "embedding",
                "source": "birdnet python package, cpu SavedModel"},
               {"sample_rate": 32000, "duration_s": 5.0, "padding": "3 s clip centre-padded to 5 s"}),
    "cnn14": ({"type": "panns_cnn14", "embedding_dim": 2048, "embedding_layer": "fc1",
               "weights": "Cnn14_mAP=0.431.pth"},
              {"sample_rate": 32000, "duration_s": 3.0}),
    "resnet": ({"type": "resnet18", "embedding_dim": 512, "weights": "torchvision IMAGENET1K_V1"},
               {"sample_rate": 16000, "duration_s": 3.0, "n_mels": 64, "n_fft": 400, "hop_length": 200,
                "scaling": "per-clip min-max to [0,1], 3 channels, ImageNet mean/std"}),
}


def save_classifier_bundle(
    classifier,
    embedding_model,
    model_name,
    species,
    training_size,
    random_seed,
    test_auc,
    n_epochs,
    results_path,
    bundle_name=None,
    labels=None,
):
    """Save complete classifier bundle for deployment.

    The embedding model file is copied for BirdNET 2.4 and Perch v8; the other encoders are
    public pretrained weights, so model_info.json records which model and layer to use.

    bundle_name: override the auto-generated directory name
                 (defaults to "{species}_{model_name}_{training_size}")
    labels:      override labels.json content; if None the default binary
                 {"0": "absent", "1": "present", "species": species} is used
    """
    if model_name not in ENCODER_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    if bundle_name is None:
        bundle_name = f"{species}_{model_name}_{training_size}"
    bundle_dir = Path(results_path) / "classifiers" / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving classifier bundle to: {bundle_dir}")

    # 1. Save trained classifier
    classifier_path = bundle_dir / "classifier.pth"
    torch.save(classifier.state_dict(), classifier_path)

    # 2. Save embedding model (or a pointer to it)
    embedding_dir = bundle_dir / "embedding_model"
    embedding_dir.mkdir(exist_ok=True)
    if model_name == "birdnet":
        shutil.copy2(MODELS / "birdnet_2.4" / "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite", embedding_dir / "model.tflite")
    elif model_name == "perch":
        for item in ["saved_model.pb", "variables", "assets"]:
            src_item = MODELS / "perch_8" / item
            if src_item.is_dir():
                shutil.copytree(src_item, embedding_dir / item, dirs_exist_ok=True)
            elif src_item.exists():
                shutil.copy2(src_item, embedding_dir / item)

    model_info, preprocessing = ENCODER_INFO[model_name]
    with open(embedding_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # 3. Save config.json
    config = {
        "species": species,
        "model_name": model_name,
        "training_size": training_size,
        "embedding_dim": model_info["embedding_dim"],
        "seed": random_seed,
        "test_auc": float(test_auc),
        "n_epochs": n_epochs,
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(bundle_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 4. Save preprocessing.json
    with open(bundle_dir / "preprocessing.json", "w") as f:
        json.dump(preprocessing, f, indent=2)

    # 5. Save labels.json
    if labels is None:
        labels = {
            "0": "absent",
            "1": "present",
            "species": species,
        }
    with open(bundle_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Classifier bundle saved successfully!")
    print(f"    - Classifier: {classifier_path.name}")
    print(f"    - Embedding model: {embedding_dir.name}/")
    print(f"    - Metadata: config.json, preprocessing.json, labels.json")
