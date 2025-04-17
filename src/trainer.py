import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch

from model_loader import *


class Trainer:
    """Handles training and evaluation using stratified K-fold."""

    def __init__(self, model_wrapper, folds):
        self.model_wrapper = model_wrapper
        self.folds = folds

    def train_and_evaluate(
        self, species, train_files, test_files, labels_train, labels_val
    ):
        """Train and evaluate the model for a single species."""
        self.model_wrapper.model.change_classes([species])
        emb_train = self.model_wrapper.embed_files(train_files)
        emb_val = self.model_wrapper.embed_files(test_files)

        self.model_wrapper.model.network.fit(
            emb_train, labels_train, emb_val, labels_val
        )
        preds = (
            self.model_wrapper.model.network(torch.tensor(emb_val, dtype=torch.float32))
            .detach()
            .numpy()
        )
        return roc_auc_score(labels_val, preds, average=None)

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

            curr_score = self.train_and_evaluate(
                species, train_files, test_files, labels_train, labels_val
            )
            roc_auc_scores.append(curr_score)
            print(f"Fold {fold_idx + 1}: ROC AUC Score = {curr_score}")

        avg_roc_auc = np.mean(roc_auc_scores)
        return avg_roc_auc
