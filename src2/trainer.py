import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def train_model(model, train_files, train_labels, device):
    """Train a classifier head on top of frozen embeddings.

    All models (BirdNET, Perch, ResNet, MobileNet, VGG) now use this
    workflow: extract embeddings, then train a small MLP head.
    """
    # Extract embeddings from all training files
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
        nn.Linear(64, 2),  # Binary classification
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


def evaluate_model(
    model, test_files, test_labels, device, original_embedding_model=None
):
    """Evaluate classifier on test set using embeddings.

    Args:
        model: The trained classifier head
        test_files: List of test audio file paths
        test_labels: List of test labels
        device: torch device
        original_embedding_model: The frozen embedding extractor
    """
    model.eval()
    predictions = []

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

    auc_score = roc_auc_score(test_labels, predictions)
    print(f"  AUC Score: {auc_score:.4f}")
    return auc_score
