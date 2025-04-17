import torch
import os

from data_loader import *


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