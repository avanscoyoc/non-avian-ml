import os
import pandas as pd
from datetime import datetime

class ResultsManager:
    """Handles saving and managing model evaluation results for individual species."""

    def __init__(self, results_path="/workspaces/non-avian-ml-toy/results"):
        """Initialize ResultsManager with results directory path."""
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)

    def generate_filename(self, model_name, species, datatype, training_size):
        """Generate standardized filename for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{species}_{model_name}_{datatype}_train{training_size}_{timestamp}.csv"

    def save_results(self, results_dict, model_name, species, datatype, training_size, batch_size, n_folds):
        """
        Save evaluation results to CSV file.
        
        Args:
            results_dict: Dictionary containing evaluation results
            model_name: Name of the model used (resnet, mobilenet, vgg, birdnet, perch)
            species: Name of the species evaluated
            datatype: Type of data used (data or data_5s)
            training_size: Number of samples per class used for training
            batch_size: Batch size used during training
            n_folds: Number of folds used in cross-validation
        """
        filename = self.generate_filename(model_name, species, datatype, training_size)
        filepath = os.path.join(self.results_path, filename)

        # Prepare results data
        results_data = {
            "model_name": model_name,
            "species": species,
            "datatype": datatype,
            "training_size": training_size,
            "batch_size": batch_size,
            "n_folds": n_folds,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mean_roc_auc": results_dict.get("mean_roc_auc", None),
        }

        # Add individual fold scores if available
        fold_scores = results_dict.get("fold_scores", {})
        for fold_idx, score in fold_scores.items():
            results_data[f"fold_{fold_idx+1}_roc_auc"] = score

        # Create and save DataFrame
        df = pd.DataFrame([results_data])
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        
        return filepath

    def load_results(self, species=None, model_name=None):
        """
        Load results from CSV files with optional filtering.
        
        Args:
            species: Optional species name to filter results
            model_name: Optional model name to filter results
        
        Returns:
            DataFrame containing all matching results
        """
        all_results = []
        for filename in os.listdir(self.results_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.results_path, filename)
                df = pd.read_csv(filepath)
                
                if species and df['species'].iloc[0] != species:
                    continue
                if model_name and df['model_name'].iloc[0] != model_name:
                    continue
                    
                all_results.append(df)
        
        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)