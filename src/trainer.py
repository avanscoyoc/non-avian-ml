import os
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch import nn, optim

from model_loader import ModelLoader
from data_processor import DataProcessor as CustomDataLoader
from save_results import ResultsManager


class BaseModelWrapper:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type  # "torch" or "bioacoustics"

    def train(self, train_data, train_labels, val_data, val_labels):
        raise NotImplementedError

    def predict(self, val_data):
        raise NotImplementedError


class TorchModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model, model_type="torch")
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters())
        self.epochs = 10
        self.device = next(model.parameters()).device

    def train_and_predict(self, train_dataset, val_dataset, batch_size):
        train_loader = TorchDataLoader(train_dataset,
                                       batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                self.optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device).float()
                    output = self.model(data)
                    val_loss += self.criterion(output.squeeze(), target).item()

            print(f'Epoch: {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}')

        # Prediction
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = torch.sigmoid(output).cpu().numpy()
                predictions.extend(pred.squeeze())

        return predictions


class BioacousticsModelWrapper(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model, model_type="bioacoustics")
        self.species = None
        self.batch_size = 32
        self.num_workers = os.cpu_count() * 3 // 4

    def embed_files(self, file_paths):
        """Generate embeddings for a list of file paths."""
        return self.model.embed(
            file_paths,
            return_dfs=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_and_predict(self, emb_train, train_labels, emb_val, val_labels):
        """Train on embeddings and predict."""
        self.model.change_classes([self.species])
        self.model.network.fit(emb_train, train_labels, emb_val, val_labels)
        preds = self.model.network(torch.tensor(emb_val, dtype=torch.float32)).detach().numpy()
        return preds


class Trainer:
    def __init__(self, model_wrapper, folds, data_loader):
        self.model_wrapper = model_wrapper
        self.folds = folds
        self.data_loader = data_loader

    def k_fold_train(self, df, species, batch_size=32):
        if isinstance(self.model_wrapper, BioacousticsModelWrapper):
            self.model_wrapper.species = species

        file_paths = df.index.values
        labels = df[species].values
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=8)
        fold_scores = []

        for train_idx, val_idx in skf.split(file_paths, labels):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            if isinstance(self.model_wrapper, BioacousticsModelWrapper):
                train_files = train_df.index.tolist()
                val_files = val_df.index.tolist()
                emb_train = self.model_wrapper.embed_files(train_files)
                emb_val = self.model_wrapper.embed_files(val_files)

                train_labels = train_df[species].values.reshape(-1, 1)
                val_labels = val_df[species].values.reshape(-1, 1)

                preds = self.model_wrapper.train_and_predict(emb_train, train_labels, emb_val, val_labels)
            else:
                train_dataset = self.data_loader.get_dataset(train_df, species, self.model_wrapper.model_type)
                val_dataset = self.data_loader.get_dataset(val_df, species, self.model_wrapper.model_type)
                preds = self.model_wrapper.train_and_predict(train_dataset, val_dataset, batch_size)

            roc_auc = roc_auc_score(val_df[species].values, preds)
            fold_scores.append(roc_auc)

        return np.mean(fold_scores), fold_scores