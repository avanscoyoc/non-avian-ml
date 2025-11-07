import os
from pathlib import Path
from config import load_config
from data_loader import (
    load_audio_files,
    create_kfold_splits,
    create_train_test_split,
)
from model_loader import load_model
from trainer import train_model, evaluate_model
from results import save_results, plot_species_models


def main():
    try:
        # Load config
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            src2_config = Path(__file__).parent / "config.yaml"
            if src2_config.exists():
                config_path = str(src2_config)

        config = load_config(config_path)
        all_results = []

        for species in config.species:
            print(f"\n{'=' * 80}\nSpecies: {species}\n{'=' * 80}")

            for model_name in config.models:
                # Determine datatype based on model
                datatype = "data_5s" if model_name == "perch" else config.datatype

                # Create fixed train/test split ONCE per species/model combo
                train_pos, train_neg, test_pos, test_neg = create_train_test_split(
                    config.data_path,
                    species,
                    datatype,
                    config.test_size_per_class,
                    config.kfold_seed,
                )

                test_files = test_pos + test_neg
                test_labels = [1] * len(test_pos) + [0] * len(test_neg)

                print(
                    f"\nModel: {model_name} | Test set: {len(test_files)} "
                    f"({len(test_pos)}+{len(test_neg)} fixed)"
                )

                for training_size in config.training_sizes:
                    print(f"  Training size: {training_size}")

                    for random_seed in config.random_seeds:
                        # Sample from train pool
                        files, labels = load_audio_files(
                            training_size, train_pos, train_neg, random_seed
                        )

                        # K-fold CV within training samples
                        fold_scores = []
                        for fold_idx, (
                            train_files,
                            train_labels,
                            val_files,
                            val_labels,
                        ) in enumerate(
                            create_kfold_splits(
                                files, labels, config.n_folds, config.kfold_seed
                            ),
                            1,
                        ):
                            # Train model
                            model, device = load_model(model_name)
                            trained_model = train_model(
                                model, train_files, train_labels, device
                            )

                            # Validate on fold
                            val_score = evaluate_model(
                                trained_model, val_files, val_labels, device, model
                            )
                            fold_scores.append(val_score)

                        # Evaluate on FIXED test set
                        final_model, device = load_model(model_name)
                        final_trained = train_model(final_model, files, labels, device)
                        test_score = evaluate_model(
                            final_trained, test_files, test_labels, device, final_model
                        )

                        # Store results
                        cv_mean = sum(fold_scores) / len(fold_scores)
                        all_results.append(
                            {
                                "species": species,
                                "model": model_name,
                                "training_size": training_size,
                                "random_seed": random_seed,
                                "cv_auc_mean": cv_mean,
                                "test_auc": test_score,
                            }
                        )
                        print(
                            f"    Seed {random_seed}: CV={cv_mean:.4f}, "
                            f"Test={test_score:.4f}"
                        )

        # Save results
        output_file = f"{config.results_path}/experiment_results.csv"
        df = save_results(all_results, output_file)
        print(f"\nResults saved to {output_file}")

        # Plot
        plot_file = output_file.replace(".csv", "_plot.png")
        plot_species_models(df, plot_file)
        print(f"Plot saved to {plot_file}")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
