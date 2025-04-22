#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torchvision import models
import onnx
import onnxruntime as ort

class AudioDataProcessor:
    """Class for processing audio files into spectrograms for ONNX models"""
    
    def __init__(self, sample_rate=44100, n_mels=80, n_fft=2048, hop_length=512):
        """
        Initialize the audio data processor
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Define mel spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
    def load_audio(self, file_path, max_length=5):
        """
        Load audio file and resample if necessary
        
        Args:
            file_path: Path to the audio file
            max_length: Maximum length in seconds
            
        Returns:
            Tensor of audio data
        """
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or trim to max_length
        max_samples = int(max_length * self.sample_rate)
        if waveform.shape[1] < max_samples:
            # Pad with zeros
            padding = torch.zeros(1, max_samples - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        elif waveform.shape[1] > max_samples:
            # Trim to max_length
            waveform = waveform[:, :max_samples]
            
        return waveform
    
    def create_spectrogram(self, waveform):
        """
        Create mel spectrogram from waveform
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Mel spectrogram tensor
        """
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to decibels
        mel_spec = T.AmplitudeToDB()(mel_spec)
        return mel_spec
    
    def normalize_spectrogram(self, spectrogram):
        """
        Normalize spectrogram values to 0-1 range
        
        Args:
            spectrogram: Mel spectrogram tensor
            
        Returns:
            Normalized spectrogram tensor
        """
        # Replace any infinity values with large finite numbers
        spectrogram = torch.nan_to_num(spectrogram, nan=0.0, posinf=80.0, neginf=-80.0)
        
        # Clip to reasonable range for audio spectrograms (in dB scale)
        spectrogram = torch.clamp(spectrogram, min=-80.0, max=80.0)
        
        # Now normalize to [0, 1] range
        min_val = torch.min(spectrogram)
        max_val = torch.max(spectrogram)
        
        # Handle case where min == max (constant spectrogram)
        if min_val == max_val:
            return torch.zeros_like(spectrogram)
            
        spectrogram = (spectrogram - min_val) / (max_val - min_val)
        
        # One final check to remove any NaNs that might have crept in
        spectrogram = torch.nan_to_num(spectrogram, nan=0.0)
        
        return spectrogram
    
    def process_file(self, file_path):
        """
        Process an audio file into a normalized spectrogram
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Normalized spectrogram tensor (3 channels for RGB compatibility)
        """
        try:
            waveform = self.load_audio(file_path)
            spectrogram = self.create_spectrogram(waveform)
            spectrogram = self.normalize_spectrogram(spectrogram)
            
            # Repeat the channel to get 3 channels (RGB compatibility with image models)
            spectrogram = spectrogram.repeat(3, 1, 1)
            
            return spectrogram
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_dataset(self, files_df):
        """
        Process all files in the dataframe
        
        Args:
            files_df: DataFrame with 'file' as index and 'present' as label
            
        Returns:
            X: Processed spectrograms
            y: Labels
        """
        X = []
        y = []
        
        for file_path, row in files_df.iterrows():
            spectrogram = self.process_file(file_path)
            if spectrogram is not None:
                X.append(spectrogram)
                y.append(1 if row['present'] else 0)
        
        return torch.stack(X), torch.tensor(y)


class AudioSpectrogramDataset(Dataset):
    """Dataset class for audio spectrograms"""
    
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


def get_model(model_name, num_classes=2):
    """
    Get a pretrained model with the classification head modified
    
    Args:
        model_name: Name of the model to use ('resnet18', 'mobilenet')
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet':
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def train_model(model, train_loader, valid_loader, device, num_epochs=10, lr=0.0001):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        
    Returns:
        Trained model, validation probabilities, validation labels
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Add weight decay for regularization
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Check  NaN in inputs
            if torch.isnan(inputs).any():
                print("Warning: NaN detected in inputs, skipping batch")
                continue
                
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in epoch {epoch+1}")
                continue
                
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch loss
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Validation phase
    model.eval()
    val_probs = []
    val_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Check for NaN in inputs
            if torch.isnan(inputs).any():
                print("Warning: NaN detected in validation inputs, skipping batch")
                continue
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of positive class
            
            # Replace any NaN values with 0.5 (neutral prediction)
            probs = np.nan_to_num(probs, nan=0.5)
            
            val_probs.extend(probs)
            val_labels.extend(labels.numpy())
    
    return model, np.array(val_probs), np.array(val_labels)


def export_to_onnx(model, dummy_input, output_path):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        dummy_input: Dummy input tensor for tracing
        output_path: Path to save the ONNX model
    """
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")


def kfold_cross_validation(data_path, species, model_name, k_folds=5, batch_size=32, epochs=10, lr=0.001):
    """
    Perform K-fold cross-validation
    
    Args:
        data_path: Path to the audio data directory
        species: Species to process
        model_name: Name of the model to use
        k_folds: Number of folds for cross-validation
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        lr: Learning rate
        
    Returns:
        Mean AUC score across all folds
    """
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_subfolder_name = "data"
    files = glob.glob(os.path.join(data_path, species, data_subfolder_name, "**/*.wav"), recursive=True)
    labels_df = pd.DataFrame({"file": files, "present": ["pos" in f.lower() for f in files]})
    labels_df['file'] = labels_df['file'].astype(str)
    labels_df.set_index("file", inplace=True)
    
    # Initialize data processor
    processor = AudioDataProcessor()
    
    # Process all data
    print(f"Processing {len(labels_df)} audio files...")
    X, y = processor.process_dataset(labels_df)
    
    # K-Fold Cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    auc_scores = []
    fold_idx = 1
    
    for train_idx, val_idx in kf.split(X):
        print(f"\nFold {fold_idx}/{k_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets and data loaders
        train_dataset = AudioSpectrogramDataset(X_train, y_train)
        val_dataset = AudioSpectrogramDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Get model
        model = get_model(model_name)
        
        # Train model
        model, val_probs, val_labels = train_model(
            model, train_loader, val_loader, device, epochs, lr
        )
        
        # Calculate AUC score
        try:
            # Ensure no NaN values
            val_probs = np.nan_to_num(val_probs, nan=0.5)
            
            # Check if we have both positive and negative examples in the validation set
            if len(np.unique(val_labels)) > 1:
                auc = roc_auc_score(val_labels, val_probs)
                print(f"Fold {fold_idx} - AUC: {auc:.4f}")
                auc_scores.append(auc)
            else:
                print(f"Fold {fold_idx} - Warning: only one class in validation set, skipping AUC calculation")
                # Use accuracy instead for this fold
                predictions = (val_probs > 0.5).astype(int)
                accuracy = np.mean(predictions == val_labels)
                print(f"Fold {fold_idx} - Accuracy: {accuracy:.4f}")
                # Use a default AUC of 0.5 (random guessing)
                auc_scores.append(0.5)
        except Exception as e:
            print(f"Error calculating AUC for fold {fold_idx}: {e}")
            print("Using default AUC of 0.5")
            auc_scores.append(0.5)
        
        # Export trained model to ONNX format
        if fold_idx == 1:  # Just save the first fold's model
            # Create dummy input with correct dimensions for the spectrogram shape
            # Using our specific mel features and typical time frames for a 5 second audio clip
            # The 3 is for RGB channels, processor.n_mels is the frequency dimension
            dummy_input = torch.randn(1, 3, processor.n_mels, 87).to(device)
            export_path = f"{species}_{model_name}_fold{fold_idx}.onnx"
            export_to_onnx(model, dummy_input, export_path)
        
        fold_idx += 1
    
    # Calculate and print average AUC
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"\nAverage AUC across {k_folds} folds: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return mean_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ONNX models for audio classification")
    parser.add_argument("--data_path", type=str, default="/workspaces/non-avian-ml-toy/data/audio",
                        help="Path to the audio data directory")
    parser.add_argument("--species", type=str, default="bullfrog",
                        help="Species to process")
    parser.add_argument("--model", type=str, choices=["resnet18", "mobilenet"], default="resnet18",
                        help="Model architecture to use")
    parser.add_argument("--folds", type=int, default=3,
                        help="Number of folds for K-fold cross-validation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    kfold_cross_validation(
        data_path=args.data_path,
        species=args.species,
        model_name=args.model,
        k_folds=args.folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    ) 