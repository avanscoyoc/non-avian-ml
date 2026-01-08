#!/usr/bin/env python3
"""
Batch inference script for running classifiers on multiple audio files.

Usage:
    python batch_inference.py <bundle_path_1> [bundle_path_2 ...] --output results.csv

Example:
    python batch_inference.py \
        results/classifiers/woodhouses_toad_birdnet_120/ \
        results/classifiers/bullfrog_birdnet_120/ \
        --output inference_results.csv
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
import sys
import csv
import argparse
from tqdm import tqdm


def load_classifier_bundle(bundle_path):
    """Load a complete classifier bundle."""
    bundle_path = Path(bundle_path)
    
    with open(bundle_path / "config.json") as f:
        config = json.load(f)
    
    with open(bundle_path / "embedding_model" / "model_info.json") as f:
        model_info = json.load(f)
    
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
        "embedding_model": embedding_model,
        "classifier": classifier,
        "bundle_path": bundle_path,
    }


def parse_filename(filename):
    """Extract species and present label from filename.
    
    Expected format: {species}-{pos|neg}-{rest}.wav
    Example: bullfrog-pos-t-11113588_143.wav
    Returns: (species, present_label)
    """
    parts = filename.split('-')
    if len(parts) < 2:
        return None, None
    
    species = parts[0]
    pos_neg = parts[1]
    
    if pos_neg == 'pos':
        present = 1
    elif pos_neg == 'neg':
        present = 0
    else:
        present = None
    
    return species, present


def collect_audio_files(data_path, ignore_data_5s=True):
    """Collect all WAV files from data/{species}/data/ folders, ignoring data_5s folders."""
    data_path = Path(data_path)
    audio_files = []
    
    for species_dir in data_path.iterdir():
        if not species_dir.is_dir():
            continue
        
        # Skip data_5s folders if requested
        if ignore_data_5s and 'data_5s' in species_dir.name:
            continue
        
        data_dir = species_dir / "data"
        if not data_dir.exists():
            continue
        
        # Also skip if the data directory itself is data_5s
        if ignore_data_5s and 'data_5s' in str(data_dir):
            continue
        
        # Collect from pos/ and neg/ subdirectories
        for subdir in ['pos', 'neg']:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                wav_files = list(subdir_path.glob("*.wav"))
                audio_files.extend(wav_files)
    
    return sorted(audio_files)


def predict_batch(bundle, audio_files, output_csv, append=False):
    """Run inference on batch of audio files and save to CSV."""
    classifier_species = bundle["config"]["species"]
    
    # Open CSV file
    mode = 'a' if append else 'w'
    file_exists = Path(output_csv).exists()
    
    with open(output_csv, mode, newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if new file or not appending
        if not append or not file_exists:
            writer.writerow(['classifier_species', 'test_file_name', 'species', 'present', 'probability_present'])
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"Processing {classifier_species}"):
            try:
                # Extract embeddings
                with torch.no_grad():
                    embeddings = bundle["embedding_model"].extract_embeddings(str(audio_file))
                    embeddings = embeddings.flatten()
                
                # Run classifier
                with torch.no_grad():
                    logits = bundle["classifier"](embeddings.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)[0]
                    prob_present = probs[1].item()
                
                # Parse filename
                filename = audio_file.name
                species, present = parse_filename(filename)
                
                if species is None or present is None:
                    print(f"  Warning: Could not parse {filename}, skipping")
                    continue
                
                # Write row
                writer.writerow([
                    classifier_species,
                    filename,
                    species,
                    present,
                    prob_present
                ])
                
            except Exception as e:
                print(f"  Error processing {audio_file.name}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='Run batch inference with classifier bundles')
    parser.add_argument('bundles', nargs='+', help='Paths to classifier bundle directories')
    parser.add_argument('--output', '-o', default='inference_results.csv', help='Output CSV file')
    parser.add_argument('--data-path', '-d', default='/workspaces/non-avian-ml/data', help='Path to data directory')
    parser.add_argument('--append', '-a', action='store_true', help='Append to existing CSV')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Batch Inference")
    print("=" * 80)
    
    # Collect all audio files
    print(f"\nCollecting audio files from: {args.data_path}")
    audio_files = collect_audio_files(args.data_path)
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("Error: No audio files found!")
        sys.exit(1)
    
    # Process each classifier bundle
    print(f"\nProcessing {len(args.bundles)} classifier bundle(s)")
    
    for i, bundle_path in enumerate(args.bundles):
        if not Path(bundle_path).exists():
            print(f"Error: Bundle not found: {bundle_path}")
            continue
        
        print(f"\n[{i+1}/{len(args.bundles)}] Loading bundle: {bundle_path}")
        bundle = load_classifier_bundle(bundle_path)
        
        print(f"  Classifier: {bundle['config']['species']} ({bundle['config']['model_name']})")
        print(f"  Training size: {bundle['config']['training_size']}")
        print(f"  Test AUC: {bundle['config']['test_auc']:.4f}")
        
        # Run predictions
        append_mode = args.append or i > 0  # Append for all after first
        predict_batch(bundle, audio_files, args.output, append=append_mode)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
