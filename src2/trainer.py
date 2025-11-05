import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import numpy as np
from data_loader import preprocess_audio
from model_loader import BirdNETModel, PerchModel


class AudioDataset(Dataset):
    def __init__(self, files, labels, is_embedding_model=False):
        self.files = files
        self.labels = labels
        self.is_embedding_model = is_embedding_model

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.is_embedding_model:
            return self.files[idx], self.labels[idx]
        else:
            return preprocess_audio(self.files[idx]), self.labels[idx]


def train_model(model, train_files, train_labels, device, is_embedding_model=False):
    if isinstance(model, (BirdNETModel, PerchModel)):
        # For embedding-based models: extract features and train classifier
        print(f"Training classifier on {len(train_files)} files using embeddings...")
        
        embeddings = []
        for i, file_path in enumerate(train_files):
            if i % 50 == 0:  # Progress indicator
                print(f"  Extracting embeddings: {i}/{len(train_files)}")
            
            # Extract embeddings using the pretrained model
            emb = model.extract_embeddings(file_path)
            embeddings.append(emb.flatten())

        # Stack embeddings into tensor
        embeddings_tensor = torch.stack(embeddings).to(device)
        labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
        
        print(f"  Embeddings shape: {embeddings_tensor.shape}")
        
        # Create classifier on top of embeddings
        embedding_dim = embeddings_tensor.shape[1]
        classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification
        ).to(device)

        # Train classifier on embeddings
        dataset = torch.utils.data.TensorDataset(embeddings_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

        classifier.train()
        for epoch in range(20):  # More epochs for embedding classifier
            epoch_loss = 0.0
            for features, labels in dataloader:
                optimizer.zero_grad()
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"    Epoch {epoch}: loss={avg_loss:.4f}")

        return classifier

    # Standard PyTorch training
    dataset = AudioDataset(train_files, train_labels, is_embedding_model)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):  # Quick training
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(
    model, test_files, test_labels, device, original_embedding_model=None
):
    model.eval()
    predictions = []

    if original_embedding_model and isinstance(
        original_embedding_model, (BirdNETModel, PerchModel)
    ):
        # For embedding model classifiers
        print(f"Evaluating on {len(test_files)} test files using embeddings...")
        
        with torch.no_grad():
            for i, file_path in enumerate(test_files):
                if i % 50 == 0:  # Progress indicator
                    print(f"  Processing: {i}/{len(test_files)}")
                
                # Extract embeddings using the original model
                emb = original_embedding_model.extract_embeddings(file_path)
                emb = emb.flatten().to(device)
                
                # Run through trained classifier
                outputs = model(emb.unsqueeze(0))
                probs = torch.softmax(outputs, dim=1)[0, 1]
                predictions.append(probs.cpu().numpy())
    else:
        # Standard PyTorch model evaluation
        dataset = AudioDataset(test_files, test_labels, False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions.extend(probs.cpu().numpy())

    auc_score = roc_auc_score(test_labels, predictions)
    print(f"  AUC Score: {auc_score:.4f}")
    return auc_score
