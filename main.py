from pathlib import Path
from src.config import load_config
from src.data_loader import (
    load_audio_files,
    create_kfold_splits,
    create_train_test_split,
)
from src.model_loader import load_model
from src.zero_shot import evaluate_zero_shot
from src.trainer import train_model, evaluate_model, cleanup_model
from src.results import save_run_result, aggregate_results, save_results, plot_species_models, save_classifier_bundle


def main():
    config_path = Path(__file__).parent / "src" / "config.yaml"
    config = load_config(str(config_path))
    Path(config.results_path).mkdir(parents=True, exist_ok=True)

    for species in config.species:
        print(f"\n{'=' * 80}\nSpecies: {species}\n{'=' * 80}")

        if config.species_sizes and species in config.species_sizes:
            test_size = config.species_sizes[species]["test"]
            training_sizes = [config.species_sizes[species]["train"]]
        else:
            test_size = config.test_size_per_class
            training_sizes = config.training_sizes

        for model_name in config.models:
            datatype = "data_5s" if model_name == "perch" else config.datatype

            train_pos, train_neg, test_pos, test_neg = create_train_test_split(
                config.data_path,
                species,
                datatype,
                test_size,
                config.kfold_seed,
            )

            test_files = test_pos + test_neg
            test_labels = [1] * len(test_pos) + [0] * len(test_neg)

            print(f"\n[{model_name.upper()}] Test set: {len(test_pos)} pos / {len(test_neg)} neg")

            for training_size in training_sizes:
                # Pre-check: skip infeasible training sizes before touching any seeds
                if len(train_pos) < training_size:
                    print(f"  Training size: {training_size} per class  [SKIP: only {len(train_pos)} pos files available after test split]")
                    continue
                if len(train_neg) < training_size:
                    print(f"  Training size: {training_size} per class  [SKIP: only {len(train_neg)} neg files available after test split]")
                    continue

                # Pre-check: skip if all seeds already completed
                all_done = all(
                    (Path(config.results_path) / f"run_{species}_{model_name}_{training_size}_{seed}.csv").exists()
                    for seed in config.random_seeds
                )
                if all_done:
                    print(f"  Training size: {training_size} per class  [SKIP: all {len(config.random_seeds)} seeds already completed]")
                    continue

                print(f"  Training size: {training_size} per class")

                for random_seed in config.random_seeds:
                    # Skip if already completed
                    run_file = Path(config.results_path) / f"run_{species}_{model_name}_{training_size}_{random_seed}.csv"
                    if run_file.exists():
                        print(f"    Seed {random_seed}: Already completed (skipping)")
                        continue
                    
                    try:
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
                            
                            del trained_model
                            cleanup_model(model)

                        final_model, device = load_model(model_name)
                        final_trained = train_model(final_model, files, labels, device, config.batch_size, config.n_epochs)
                        test_score = evaluate_model(
                            final_trained, test_files, test_labels, device, final_model
                        )

                        # Save classifier bundle if enabled
                        if config.save_classifier:
                            save_classifier_bundle(
                                classifier=final_trained,
                                embedding_model=final_model,
                                model_name=model_name,
                                species=species,
                                training_size=training_size,
                                random_seed=random_seed,
                                test_auc=test_score,
                                n_epochs=config.n_epochs,
                                results_path=config.results_path,
                            )

                        cv_mean = sum(fold_scores) / len(fold_scores)
                        result = {
                            "species": species,
                            "model": model_name,
                            "training_size": training_size,
                            "n_folds": config.n_folds,
                            "n_epochs": config.n_epochs,
                            "batch_size": config.batch_size,
                            "test_size_per_class": test_size,
                            "random_seed": random_seed,
                            "cv_auc_mean": cv_mean,
                            "test_auc": test_score,
                        }
                        save_run_result(result, config.results_path)
                        print(f"    Seed {random_seed}: CV AUC={cv_mean:.4f} | Test AUC={test_score:.4f}")
                        
                        del final_trained
                        cleanup_model(final_model)
                    
                    except Exception as e:
                        print(f"    ERROR Seed {random_seed}: {e}")
                        continue

            # Run zero-shot evaluation AFTER all training sizes/seeds complete
            if config.run_zeroshot:
                print(f"\n[{model_name.upper()}] Running zero-shot evaluation...")
                evaluate_zero_shot(species, model_name, test_files, test_labels, config)
            
        # Aggregate all runs for this species
        print(f"\nAggregating results for {species}...")
        all_runs = aggregate_results(config.results_path, species)
        if all_runs is not None:
            output_file = f"{config.results_path}/results_{species}.csv"
            df = save_results(all_runs.to_dict('records'), output_file)
            print(f"Results saved: {output_file}")

            figs_dir = Path(config.results_path).parent / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            plot_file = str(figs_dir / f"results_{species}_plot.png")
            plot_species_models(df, plot_file)
            print(f"Plot saved: {plot_file}")


if __name__ == "__main__":
    main()
