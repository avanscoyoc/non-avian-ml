import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from model_loader import * 


class Trainer:
    """Handles training and evaluation using stratified K-fold."""

    def __init__(self, model_wrapper, folds):
        self.model_wrapper = model_wrapper
        self.folds = folds

    def train_and_evaluate_species(self, df, species):
        """Train and evaluate for a single species."""
        print(f"Processing species: {species}")
        file_paths = df.index
        labels = df[species]
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=8)
        roc_auc_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(file_paths, labels)):
            train_files = file_paths[train_idx].tolist()
            test_files = file_paths[test_idx].tolist()
            labels_train = labels.iloc[train_idx].to_numpy().reshape(-1, 1)
            labels_val = labels.iloc[test_idx].to_numpy().reshape(-1, 1)

            curr_score = self.model_wrapper.train_and_evaluate(
                species, train_files, test_files, labels_train, labels_val
            )
            roc_auc_scores.append(curr_score)
            print(f"Fold {fold_idx + 1}: ROC AUC Score = {curr_score}")

        avg_roc_auc = np.mean(roc_auc_scores)
        return avg_roc_auc


class ResultManager:
    """Handles saving and storing model evaluation results."""

    def __init__(self, base_path="results"):
        """Initialize ResultManager with base path for results."""
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def generate_filename(
        self, model_name, species_list, training_size, random_seed, batch_size, folds
    ):
        """Generate standardized filename for results."""
        species_str = "_".join(species_list)
        training_size_str = str(training_size) if training_size else "all"
        return f"{model_name}_{species_str}_train{training_size_str}_seed{random_seed}_batch{batch_size}_fold{folds}.csv"

    def save_results(
        self,
        results,
        model_name,
        species_list,
        training_size,
        random_seed,
        batch_size,
        folds,
    ):
        """Save results to CSV file with standardized format."""
        filename = self.generate_filename(
            model_name, species_list, training_size, random_seed, batch_size, folds
        )
        filepath = os.path.join(self.base_path, filename)

        # Create results DataFrame
        results_data = []
        for species, avg_roc_auc in results.items():
            results_data.append(
                {
                    "model_name": model_name,
                    "species": species,
                    "training_size": training_size,
                    "avg_roc_auc": avg_roc_auc,
                    "random_seed": random_seed,
                    "batch_size": batch_size,
                    "folds": folds,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        df = pd.DataFrame(results_data)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        return filepath

def run_model(
    datapath,
    species_list,
    datatype,
    model_name,
    batch_size,
    folds,
    training_size=None,
    random_seed=42,
    results_path="results",
):
    """Main function to train and evaluate species with optional random sampling."""
    # Load data
    data_loader = DataLoader(
        datapath, species_list, datatype, training_size, random_seed
    )

    try:
        df = data_loader.load_data()
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Skipping this run due to insufficient samples.")
        return None

    # Initialize model and trainer
    model_wrapper = ModelWrapper(model_name, batch_size)
    trainer = Trainer(model_wrapper, folds)

    # Train and evaluate for each species
    results = {}
    for species in species_list:
        avg_roc_auc = trainer.train_and_evaluate_species(df, species)
        results[species] = avg_roc_auc

    # Save results only if we have valid results
    if results:
        result_manager = ResultManager(results_path)
        result_manager.save_results(
            results,
            model_name,
            species_list,
            training_size,
            random_seed,
            batch_size,
            folds,
        )
    
    if results is None:
        print(
            "Run was skipped due to insufficient samples. No results file was created."
        )
    else:
        print("Run completed successfully.")

    return results

