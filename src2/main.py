import random
from config import load_config
from data_loader import load_audio_files, create_kfold_splits
from model_loader import load_model
from trainer import train_model, evaluate_model
from results import save_results


def main():
    try:
        config = load_config()
        all_results = []

        for species in config.species:
            for model_name in config.models:
                for training_size in config.training_sizes:
                    print(
                        f"Running {model_name} on {species} with training_size={training_size}"
                    )

                    # Set seed
                    random.seed(config.random_seed)

                    # Load data
                    datatype = "data_5s" if model_name == "perch" else "data"
                    files, labels = load_audio_files(
                        config.data_path, species, training_size, datatype
                    )
                    print(f"Loaded {len(files)} files")

                    # K-fold CV
                    splits = create_kfold_splits(files, labels, config.n_folds)
                    fold_scores = []

                    for fold, (train_idx, val_idx) in enumerate(splits):
                        print(f"  Fold {fold + 1}/{config.n_folds}")

                        train_files = [files[i] for i in train_idx]
                        train_labels = [labels[i] for i in train_idx]
                        val_files = [files[i] for i in val_idx]
                        val_labels = [labels[i] for i in val_idx]

                        # Load and train model
                        original_model, device = load_model(model_name)
                        is_embedding = model_name in ["birdnet", "perch"]

                        trained_model = train_model(
                            original_model,
                            train_files,
                            train_labels,
                            device,
                            is_embedding,
                        )

                        # Evaluate
                        if is_embedding:
                            score = evaluate_model(
                                trained_model,
                                val_files,
                                val_labels,
                                device,
                                original_model,
                            )
                        else:
                            score = evaluate_model(
                                trained_model, val_files, val_labels, device
                            )
                        fold_scores.append(score)
                        print(f"    Fold {fold + 1} score: {score:.4f}")

                    # Save results
                    mean_score = sum(fold_scores) / len(fold_scores)
                    result = {
                        "species": species,
                        "model": model_name,
                        "training_size": training_size,
                        "mean_auc": mean_score,
                        "fold_scores": fold_scores,
                    }
                    all_results.append(result)
                    print(f"  Mean AUC: {mean_score:.4f}")

        # Save all results
        output_file = f"{config.results_path}/experiment_results.csv"
        save_results(all_results, output_file)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
