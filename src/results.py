import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import json
import shutil
from datetime import datetime


def save_run_result(result, results_path):
    """Save single run result immediately to CSV."""
    output_file = Path(results_path) / f"run_{result['species']}_{result['model']}_{result['training_size']}_{result['random_seed']}.csv"
    pd.DataFrame([result]).to_csv(output_file, index=False)


def aggregate_results(results_path, species):
    """Load all run CSVs for a species and aggregate statistics."""
    run_files = sorted(Path(results_path).glob(f"run_{species}_*.csv"))
    if not run_files:
        return None
    
    all_runs = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
    return all_runs


def save_results(results, output_path):
    """Aggregate results across random seeds and save to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Calculate aggregated statistics
    agg_results = df.groupby([
        "model", "species", "training_size", 
        "n_folds", "n_epochs", "batch_size", "test_size_per_class"
    ]).agg({
        "cv_auc_mean": ["mean", "std", "count"],
        "test_auc": ["mean", "std", "count"],
    }).round(4)

    # Flatten column names
    agg_results.columns = [
        "cv_auc_mean", "cv_auc_std", "cv_auc_n",
        "test_auc_mean", "test_auc_std", "test_auc_n",
    ]
    agg_results = agg_results.reset_index()
    
    # Calculate 95% CI: 1.96 * std / sqrt(n)
    agg_results["cv_auc_ci_95"] = (
        1.96 * agg_results["cv_auc_std"] / np.sqrt(agg_results["cv_auc_n"])
    ).round(4)
    agg_results["test_auc_ci_95"] = (
        1.96 * agg_results["test_auc_std"] / np.sqrt(agg_results["test_auc_n"])
    ).round(4)
    
    # Reorder columns for better readability
    column_order = [
        "model", "species", "training_size", 
        "n_folds", "n_epochs", "batch_size", "test_size_per_class",
        "cv_auc_mean", "cv_auc_std", "cv_auc_ci_95", "cv_auc_n",
        "test_auc_mean", "test_auc_std", "test_auc_ci_95", "test_auc_n",
    ]
    agg_results = agg_results[column_order]
    
    agg_results.to_csv(output_path, index=False)
    return agg_results


def plot_species_models(df, output_path):
    """Plot learning curves with 95% confidence intervals."""
    species_list = df["species"].unique()
    n_species = len(species_list)
    fig, axes = plt.subplots(1, n_species, figsize=(6 * n_species, 6))
    if n_species == 1:
        axes = [axes]

    for i, species in enumerate(species_list):
        species_data = df[df["species"] == species]
        models = species_data["model"].unique()

        for model in models:
            model_data = species_data[species_data["model"] == model]
            axes[i].errorbar(
                model_data["training_size"],
                model_data["test_auc_mean"],
                yerr=model_data["test_auc_ci_95"],
                label=model.upper(),
                marker="o",
                capsize=5,
                capthick=2,
            )

        axes[i].set_title(f"{species.replace('_', ' ').title()}")
        axes[i].set_xlabel("Training Size (samples per class)")
        axes[i].set_ylabel("Test ROC-AUC")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    bundle_name: override the auto-generated directory name
                 (defaults to "{species}_{model_name}_{training_size}")
    labels:      override labels.json content; if None the default binary
                 {"0": "absent", "1": "present", "species": species} is used
    """
    if bundle_name is None:
        bundle_name = f"{species}_{model_name}_{training_size}"
    bundle_dir = Path(results_path) / "classifiers" / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving classifier bundle to: {bundle_dir}")
    
    # 1. Save trained classifier
    classifier_path = bundle_dir / "classifier.pth"
    torch.save(classifier.state_dict(), classifier_path)
    
    # 2. Save embedding model
    embedding_dir = bundle_dir / "embedding_model"
    embedding_dir.mkdir(exist_ok=True)
    
    if model_name == "birdnet":
        # Copy BirdNET TFLite model
        src_model = Path("/workspaces/non-avian-ml/models/birdnet_2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
        dst_model = embedding_dir / "model.tflite"
        shutil.copy2(src_model, dst_model)
        
        model_info = {
            "type": "birdnet",
            "embedding_dim": 6522,
        }
        
        preprocessing = {
            "sample_rate": 48000,
            "duration_s": 3.0,
        }
        
    elif model_name == "perch":
        # Copy Perch SavedModel directory
        src_model_dir = Path("/workspaces/non-avian-ml/models/perch_8")
        for item in ["saved_model.pb", "variables", "assets"]:
            src_item = src_model_dir / item
            if src_item.exists():
                if src_item.is_dir():
                    shutil.copytree(src_item, embedding_dir / item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, embedding_dir / item)
        
        model_info = {
            "type": "perch",
            "embedding_dim": 1280,
        }
        
        preprocessing = {
            "sample_rate": 32000,
            "duration_s": 5.0,
        }
        
    elif model_name in ["resnet", "mobilenet", "vgg"]:
        # Save frozen CNN state_dict
        torch.save(embedding_model.state_dict(), embedding_dir / "embedding_model.pth")
        
        embedding_dims = {"resnet": 512, "mobilenet": 1280, "vgg": 4096}
        model_info = {
            "type": model_name,
            "embedding_dim": embedding_dims[model_name],
        }
        
        preprocessing = {
            "sample_rate": 16000,
            "duration_s": 3.0,
            "n_mels": 64,
            "n_fft": 2048,
            "hop_length": 512,
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Save model_info.json
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
