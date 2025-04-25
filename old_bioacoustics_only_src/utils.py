import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


from trainer import *

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
