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
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    all_results = []

    for species in config.species:
        print(f"\n{'=' * 80}\nSpecies: {species}\n{'=' * 80}")

        for model_name in config.models:
            datatype = "data_5s" if model_name == "perch" else config.datatype

            train_pos, train_neg, test_pos, test_neg = create_train_test_split(
                config.data_path,
                species,
                datatype,
                config.test_size_per_class,
                config.kfold_seed,
            )

            test_files = test_pos + test_neg
            test_labels = [1] * len(test_pos) + [0] * len(test_neg)

            print(f"\n[{model_name.upper()}] Test set: {len(test_pos)} pos / {len(test_neg)} neg")

            for training_size in config.training_sizes:
                print(f"  Training size: {training_size} per class")

                for random_seed in config.random_seeds:
                    files, labels = load_audio_files(
                        training_size, train_pos, train_neg, random_seed
                    )

                    fold_scores = []
                    for train_files, train_labels, val_files, val_labels in create_kfold_splits(
                        files, labels, config.n_folds, config.kfold_seed
                    ):
                        model, device = load_model(model_name)
                        trained_model = train_model(
                            model, train_files, train_labels, device, config.batch_size, config.n_epochs
                        )
                        val_score = evaluate_model(
                            trained_model, val_files, val_labels, device, model
                        )
                        fold_scores.append(val_score)

                    final_model, device = load_model(model_name)
                    final_trained = train_model(final_model, files, labels, device, config.batch_size, config.n_epochs)
                    test_score = evaluate_model(
                        final_trained, test_files, test_labels, device, final_model
                    )

                    cv_mean = sum(fold_scores) / len(fold_scores)
                    all_results.append(
                        {
                            "species": species,
                            "model": model_name,
                            "training_size": training_size,
                            "n_folds": config.n_folds,
                            "n_epochs": config.n_epochs,
                            "batch_size": config.batch_size,
                            "test_size_per_class": config.test_size_per_class,
                            "random_seed": random_seed,
                            "cv_auc_mean": cv_mean,
                            "test_auc": test_score,
                        }
                    )
                    print(f"    Seed {random_seed}: CV AUC={cv_mean:.4f} | Test AUC={test_score:.4f}")

    output_file = f"{config.results_path}/results_{species}.csv"
    df = save_results(all_results, output_file)
    print(f"\nResults saved: {output_file}")

    plot_file = output_file.replace(".csv", "_plot.png")
    plot_species_models(df, plot_file)
    print(f"Plot saved: {plot_file}")


if __name__ == "__main__":
    main()
