#!/usr/bin/env python3
# Test file for the bullfrog model 
import os
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort

class AudioProcessor:
    """Class for processing audio files into spectrograms for ONNX models"""
    
    def __init__(self, sample_rate=44100, n_mels=80, n_fft=2048, hop_length=512):
        """Initialize the audio processor"""
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
        """Load audio file and resample if necessary"""
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
        """Create mel spectrogram from waveform"""
        mel_spec = self.mel_spectrogram(waveform)
        # Convert to decibels
        mel_spec = T.AmplitudeToDB()(mel_spec)
        return mel_spec
    
    def normalize_spectrogram(self, spectrogram):
        """Normalize spectrogram values to 0-1 range"""
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
        """Process an audio file into a normalized spectrogram"""
        try:
            waveform = self.load_audio(file_path)
            spectrogram = self.create_spectrogram(waveform)
            spectrogram = self.normalize_spectrogram(spectrogram)
            
            # Get the current dimensions
            _, height, width = spectrogram.shape
            
            # The model expects a time dimension of 87 frames
            target_width = 87
            
            # Resize the time dimension (width) to match the expected size
            if width > target_width:
                # Trim to target_width
                spectrogram = spectrogram[:, :, :target_width]
            elif width < target_width:
                # Pad with zeros to reach target_width
                padding = torch.zeros(1, height, target_width - width)
                spectrogram = torch.cat([spectrogram, padding], dim=2)
            
            # Repeat the channel to get 3 channels (for RGB compatibility with image models)
            spectrogram = spectrogram.repeat(3, 1, 1)
            
            return spectrogram
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def test_audio(model_path, audio_path):
    """Test a single audio file with the ONNX model"""
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    # Create ONNX Runtime session
    try:
        session = ort.InferenceSession(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
    
    # Process the audio file
    processor = AudioProcessor()
    print(f"Processing audio file: {audio_path}")
    spectrogram = processor.process_file(audio_path)
    
    if spectrogram is None:
        print(f"Failed to process audio file")
        return
    
    # Prepare input for the model (add batch dimension)
    input_data = spectrogram.unsqueeze(0).numpy()
    
    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"Model input name: {input_name}")
    
    # Run inference
    print("Running inference...")
    outputs = session.run(None, {input_name: input_data})
    
    # Get probabilities using softmax
    logits = outputs[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    # Get prediction for positive class (index 1)
    prediction = probabilities[0, 1]
    
    # Print result
    print("\nPrediction Results:")
    print(f"File: {os.path.basename(audio_path)}")
    is_positive = prediction > 0.5
    confidence = prediction if is_positive else 1 - prediction
    print(f"Class: {'Positive (bullfrog)' if is_positive else 'Negative (not bullfrog)'}")
    print(f"Confidence: {confidence:.4f}")
    
    return prediction

def batch_test(model_path, directory, recursive=True):
    """Test all audio files in a directory"""
    import glob
    
    # Find all audio files
    pattern = os.path.join(directory, "**/*.wav" if recursive else "*.wav")
    files = glob.glob(pattern, recursive=recursive)
    
    if not files:
        print(f"No audio files found in {directory}")
        return
    
    print(f"Found {len(files)} audio files")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create ONNX Runtime session
    try:
        session = ort.InferenceSession(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
    
    # Process all files
    processor = AudioProcessor()
    
    results = {
        'positive': [],
        'negative': []
    }
    
    for i, file_path in enumerate(files):
        print(f"\nProcessing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
        
        # Process the audio file
        spectrogram = processor.process_file(file_path)
        
        if spectrogram is None:
            print(f"  Failed to process - skipping")
            continue
        
        # Prepare input for the model
        input_data = spectrogram.unsqueeze(0).numpy()
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Get prediction
        logits = outputs[0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        prediction = probabilities[0, 1]
        
        # Store result
        is_positive = prediction > 0.5
        confidence = prediction if is_positive else 1 - prediction
        
        category = 'positive' if is_positive else 'negative'
        results[category].append((file_path, confidence))
        
        # Print result
        expected = "pos" if "pos" in file_path.lower() else "neg" if "neg" in file_path.lower() else "unknown"
        correct = (expected == "pos" and is_positive) or (expected == "neg" and not is_positive)
        status = "✓" if (expected != "unknown" and correct) else "✗" if (expected != "unknown" and not correct) else "-"
        
        print(f"  Prediction: {'Positive' if is_positive else 'Negative'} (confidence: {confidence:.4f}) {status}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Positive predictions: {len(results['positive'])}/{len(files)}")
    print(f"  Negative predictions: {len(results['negative'])}/{len(files)}")
    
    # If there are any files with "pos" or "neg" in the name, we can calculate accuracy
    true_positives = sum(1 for path, _ in results['positive'] if "pos" in path.lower())
    false_positives = sum(1 for path, _ in results['positive'] if "neg" in path.lower())
    true_negatives = sum(1 for path, _ in results['negative'] if "neg" in path.lower())
    false_negatives = sum(1 for path, _ in results['negative'] if "pos" in path.lower())
    
    labeled_files = true_positives + false_positives + true_negatives + false_negatives
    
    if labeled_files > 0:
        accuracy = (true_positives + true_negatives) / labeled_files
        print(f"\nAccuracy on labeled files: {accuracy:.4f}")
        
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            print(f"Recall (true positive rate): {recall:.4f}")
        
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            print(f"Precision: {precision:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test audio with ONNX bullfrog model")
    parser.add_argument("--model", type=str, default="bullfrog_resnet18_fold1.onnx", 
                        help="Path to ONNX model")
    parser.add_argument("--batch", action="store_true", 
                        help="Test a batch of files instead of a single file")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to audio file or directory (if --batch is used)")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_test(args.model, args.input)
    else:
        test_audio(args.model, args.input)

if __name__ == "__main__":
    main() 