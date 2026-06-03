# Classifier Bundle Documentation

## Overview

A classifier bundle is a self-contained deployment package that includes everything needed to run inference on audio files for a specific species using a trained model.

```
pixi run python load_classifier_example.py results/classifiers/woodhouses_toad_birdnet_120/ test_audio/woodhousestoad_26sec.wav
```
## Bundle Structure

Each bundle is organized as follows:

```
{species}_{model}_{training_size}/
├── embedding_model/              # Frozen feature extractor
│   ├── model.tflite             # BirdNET (40MB)
│   │   OR saved_model.pb        # Perch (150MB) + variables/ + assets/
│   │   OR embedding_model.pth   # CNNs (45MB)
│   └── model_info.json          # Model metadata
├── classifier.pth                # Trained MLP classifier (450KB)
├── config.json                   # Bundle configuration
├── preprocessing.json            # Audio preprocessing parameters
└── labels.json                   # Output class labels
```

## Configuration Files

### config.json
Contains bundle metadata and training information:
```json
{
  "species": "bullfrog",
  "model_name": "birdnet",
  "training_size": 100,
  "embedding_dim": 6522,
  "seed": 1,
  "test_auc": 0.94,
  "n_epochs": 20,
  "created_date": "2026-01-08 12:34:56"
}
```

### preprocessing.json
Audio preprocessing parameters (varies by model):
```json
{
  "sample_rate": 48000,    // BirdNET=48kHz, Perch=32kHz, CNNs=16kHz
  "duration_s": 3.0,       // BirdNET/CNNs=3s, Perch=5s
  "n_mels": 64,            // CNNs only
  "n_fft": 2048,           // CNNs only
  "hop_length": 512        // CNNs only
}
```

### labels.json
Output class mapping:
```json
{
  "0": "absent",
  "1": "present",
  "species": "bullfrog"
}
```

### model_info.json
Embedding model metadata:
```json
{
  "type": "birdnet",       // "birdnet", "perch", "resnet", "mobilenet", "vgg"
  "embedding_dim": 6522
}
```

## Creating Bundles

To save classifier bundles during training:

1. Edit `src/config.yaml`:
   ```yaml
   save_classifier: true
   ```

2. Run the experiment:
   ```bash
   pixi run python src/main.py
   ```

3. Bundles will be saved to:
   ```
   results/classifiers/{species}_{model}_{training_size}/
   ```

## Using Bundles for Inference

### Example Usage

```python
from load_classifier_example import load_classifier_bundle, predict

# Load bundle
bundle = load_classifier_bundle("results/classifiers/bullfrog_birdnet_100/")

# Run inference
result = predict(bundle, "mystery_audio.wav")

print(f"Species: {result['species']}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Command Line

```bash
python load_classifier_example.py \
    results/classifiers/bullfrog_birdnet_100/ \
    test_audio.wav
```

## Inference Pipeline

The complete inference pipeline:

```
Audio File (WAV)
    ↓
1. Load & Resample (to model's sample_rate)
    ↓
2. Preprocess (CNNs: → mel-spectrogram, BirdNET/Perch: → waveform chunk)
    ↓
3. Extract Embeddings (frozen model → high-dim vector)
    ↓
4. Classify (MLP → probabilities)
    ↓
Output: {prediction: "present", confidence: 0.87}
```

## Dependencies

### Required Python Packages

- **PyTorch** (`torch`, `torchaudio`, `torchvision`) - For CNN models and classifier
- **TensorFlow** (`tensorflow`, `tensorflow_hub`) - For Perch model
- **ai-edge-litert** - For BirdNET TFLite model
- **librosa** - Audio processing
- **soundfile** - Audio I/O for Perch
- **numpy** - Numerical operations

### Installation

```bash
# Using pixi (recommended)
pixi install

# Or using pip
pip install torch torchaudio torchvision tensorflow tensorflow_hub \
    ai-edge-litert librosa soundfile numpy
```

## Bundle Sizes

Approximate bundle sizes by model:

| Model | Embedding Model | Classifier | Total |
|-------|----------------|------------|-------|
| BirdNET | 40 MB | 450 KB | ~42 MB |
| Perch | 150 MB | 450 KB | ~152 MB |
| ResNet18 | 45 MB | 450 KB | ~47 MB |
| MobileNetV2 | 45 MB | 450 KB | ~47 MB |
| VGG11 | 45 MB | 450 KB | ~47 MB |

## Integration with Web Platform

For web deployment, each bundle provides:

1. **Self-contained models** - No external dependencies on training data or code
2. **JSON metadata** - Easy parsing in any language
3. **Standard PyTorch format** - Compatible with TorchServe, ONNX, etc.
4. **Preprocessing specs** - Exact parameters for audio preparation

### Deployment Checklist

- [ ] Copy entire bundle directory to deployment server
- [ ] Install required dependencies (see above)
- [ ] Load bundle using `load_classifier_bundle()`
- [ ] Implement audio preprocessing pipeline
- [ ] Run inference on uploaded audio files
- [ ] Return predictions as JSON API response

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'yaml'"**
- Install PyYAML: `pip install pyyaml`

**"Unable to resolve nonexistent file"**
- Ensure you're using absolute paths or working from project root

**"Embedding dimension mismatch"**
- Classifier was trained on different model version
- Regenerate bundle with correct model

**"Audio file too short"**
- Pad audio to minimum duration (3s for BirdNET/CNNs, 5s for Perch)
- Or split longer audio into chunks

## Version Compatibility

- **PyTorch**: ≥ 1.12
- **TensorFlow**: ≥ 2.15
- **Python**: 3.10+

## Support

For issues or questions, refer to the main project README or documentation.
