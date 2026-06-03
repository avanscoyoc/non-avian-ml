#!/usr/bin/env python3
"""
Example script showing how to load and use a saved classifier bundle for inference.

Usage:
    python scripts/load_classifier_example.py <bundle_path> <audio_file>

Example:
    python scripts/load_classifier_example.py results/classifiers/woodhouses_toad_birdnet_511/ test_audio/woodhousestoad_26sec.wav
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
import sys


def load_classifier_bundle(bundle_path):
    """Load a complete classifier bundle."""
    bundle_path = Path(bundle_path)
    
    # Load configuration files
    with open(bundle_path / "config.json") as f:
        config = json.load(f)
    
    with open(bundle_path / "preprocessing.json") as f:
        preprocessing = json.load(f)
    
    with open(bundle_path / "labels.json") as f:
        labels = json.load(f)
    
    with open(bundle_path / "embedding_model" / "model_info.json") as f:
        model_info = json.load(f)
    
    print(f"Loaded bundle: {config['species']} - {config['model_name']}")
    print(f"  Training size: {config['training_size']}")
    print(f"  Test AUC: {config['test_auc']:.4f}")
    print(f"  Embedding dim: {model_info['embedding_dim']}")
    
    # Load embedding model
    model_type = model_info['type']
    embedding_dir = bundle_path / "embedding_model"
    
    if model_type == "birdnet":
        from src.model_loader import BirdNETModel
        model_path = str(embedding_dir / "model.tflite")
        embedding_model = BirdNETModel(model_path)
        
    elif model_type == "perch":
        from src.model_loader import PerchModel
        embedding_model = PerchModel(str(embedding_dir))
        
    elif model_type in ["resnet", "mobilenet", "vgg"]:
        from src.model_loader import CNNEmbeddingModel
        embedding_model = CNNEmbeddingModel(model_type)
        embedding_model.load_state_dict(torch.load(embedding_dir / "embedding_model.pth"))
        embedding_model.eval()
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load trained classifier
    embedding_dim = model_info['embedding_dim']
    classifier = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2),
    )
    classifier.load_state_dict(torch.load(bundle_path / "classifier.pth"))
    classifier.eval()
    
    return {
        "config": config,
        "preprocessing": preprocessing,
        "labels": labels,
        "embedding_model": embedding_model,
        "classifier": classifier,
    }


def predict(bundle, audio_file):
    """Run inference on an audio file."""
    print(f"\nRunning inference on: {audio_file}")
    
    # Extract embeddings
    print("  Extracting embeddings...")
    with torch.no_grad():
        embeddings = bundle["embedding_model"].extract_embeddings(audio_file)
        embeddings = embeddings.flatten()
    
    # Run classifier
    print("  Running classifier...")
    with torch.no_grad():
        logits = bundle["classifier"](embeddings.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
    
    prob_absent = probs[0].item()
    prob_present = probs[1].item()
    
    # Get prediction
    predicted_class = 1 if prob_present > 0.5 else 0
    prediction = bundle["labels"][str(predicted_class)]
    confidence = max(prob_absent, prob_present)
    
    result = {
        "species": bundle["config"]["species"],
        "prediction": prediction,
        "confidence": confidence,
        "probability_absent": prob_absent,
        "probability_present": prob_present,
    }
    
    return result


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    bundle_path = sys.argv[1]
    audio_file = sys.argv[2]
    
    if not Path(bundle_path).exists():
        print(f"Error: Bundle path not found: {bundle_path}")
        sys.exit(1)
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("Classifier Bundle Inference Example")
    print("=" * 70)
    print()
    
    # Load bundle
    bundle = load_classifier_bundle(bundle_path)
    
    # Run prediction
    result = predict(bundle, audio_file)
    
    # Display results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Species: {result['species']}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
    print(f"Probability absent:  {result['probability_absent']:.4f}")
    print(f"Probability present: {result['probability_present']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()