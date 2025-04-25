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
