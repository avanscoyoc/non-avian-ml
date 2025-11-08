import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


def train_model(model, train_files, train_labels, device, batch_size=32, n_epochs=20):
    """Train MLP classifier on frozen embeddings."""
    print(f"  Extracting embeddings: {len(train_files)} files")

    embeddings = []
    for i, file_path in enumerate(train_files):
        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(train_files)}")
        emb = model.extract_embeddings(file_path)
        embeddings.append(emb.flatten())

    embeddings_tensor = torch.stack(embeddings).to(device)
    labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)

    print(f"  Training classifier: {embeddings_tensor.shape}")

    embedding_dim = embeddings_tensor.shape[1]
    classifier = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    ).to(device)

    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    classifier.train()
    for epoch in range(n_epochs):
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


def evaluate_model(model, test_files, test_labels, device, original_embedding_model):
    """Evaluate classifier on test set using embeddings."""
    model.eval()
    predictions = []

    print(f"  Evaluating: {len(test_files)} files")

    with torch.no_grad():
        for i, file_path in enumerate(test_files):
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i + 1}/{len(test_files)}")

            emb = original_embedding_model.extract_embeddings(file_path)
            emb = emb.flatten().to(device)

            outputs = model(emb.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)[0, 1]
            predictions.append(probs.cpu().numpy())

    auc_score = roc_auc_score(test_labels, predictions)
    print(f"  AUC: {auc_score:.4f}")
    return auc_score
