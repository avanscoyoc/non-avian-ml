import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random
from datetime import datetime


class DataLoader:
    """Handles loading and preprocessing of data."""

    def __init__(
        self, datapath, species_list, datatype, training_size=None, random_seed=42
    ):
        self.datapath = datapath
        self.species_list = species_list
        self.datatype = datatype
        self.training_size = training_size
        random.seed(random_seed)

    def sample_files(self, files, size, species, file_type="positive"):
        """Helper method to sample files with proper error handling."""
        if len(files) < size:
            print(
                f"Warning: Requested {size} {file_type} samples but only found {len(files)} for {species}"
            )
            return files
        return random.sample(files, size)

    def load_species_data(self, species):
        """Load data for a single species with optional random sampling."""
        pos_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
        )
        neg_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")
        )

        # Check if we have enough samples before proceeding
        if self.training_size is not None:
            if (
                len(pos_files) < self.training_size
                or len(neg_files) < self.training_size
            ):
                raise ValueError(
                    f"Insufficient samples for {species}: "
                    f"Found {len(pos_files)} positive and {len(neg_files)} negative samples, "
                    f"but {self.training_size} samples were requested."
                )

        # Random sampling if training_size is specified
        if self.training_size is not None:
            min_samples = min(len(pos_files), len(neg_files))
            training_size = min(self.training_size, min_samples)
            pos_files = self.sample_files(pos_files, training_size, species, "positive")
            neg_files = self.sample_files(neg_files, training_size, species, "negative")

            print(
                f"Using {len(pos_files)} positive and {len(neg_files)} negative samples for {species}"
            )

        all_files = pos_files + neg_files
        encoding_pos_files = [1] * len(pos_files) + [0] * len(neg_files)
        encoding_neg_files = [0] * len(pos_files) + [1] * len(neg_files)

        return pd.DataFrame(
            {
                "files": all_files,
                species: encoding_pos_files,
                "noise": encoding_neg_files,
            }
        )

    def load_data(self):
        """Load data for all species."""
        print("Loading dataset...")
        df_each_species = {}
        for species in self.species_list:
            df_each_species[species] = self.load_species_data(species)

        all_species = pd.concat(df_each_species.values(), axis=0)
        all_species.fillna(0, inplace=True)
        all_species.set_index("files", inplace=True)
        all_species = all_species.astype(int)
        print("Dataset loaded:")
        print(all_species.sum("index"))
        print(all_species.head())
        return all_species


class ModelWrapper:
    """Encapsulates model loading, embedding, and training."""

    def __init__(self, model_name, batch_size):
        print("Loading model...")
        self.model = torch.hub.load(
            "kitzeslab/bioacoustics-model-zoo", model_name, trust_repo=True
        )
        self.num_workers = os.cpu_count() * 3 // 4  # Use 75% of CPU cores
        self.batch_size = batch_size
        print(f"CPU CORES: {self.num_workers}")
        print(f"Model loaded: {model_name}")

    def embed_files(self, file_paths):
        """Generate embeddings for a list of file paths."""
        return self.model.embed(
            file_paths,
            return_dfs=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_and_evaluate(
        self, species, train_files, test_files, labels_train, labels_val
    ):
        """Train and evaluate the model for a single species."""
        self.model.change_classes([species])
        emb_train = self.embed_files(train_files)
        emb_val = self.embed_files(test_files)

        self.model.network.fit(emb_train, labels_train, emb_val, labels_val)
        preds = (
            self.model.network(torch.tensor(emb_val, dtype=torch.float32))
            .detach()
            .numpy()
        )
        return roc_auc_score(labels_val, preds, average=None)


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


def main(
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

    return results
