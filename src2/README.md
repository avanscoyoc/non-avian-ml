# Non-Avian ML Audio Classification Experiment Framework (src2)

## Overview

This application is a modular machine learning experiment framework designed for comparing audio classification models across different species and training data sizes. The system extracts embeddings from pretrained models and trains custom classifiers for species-specific binary classification tasks.

## Key Features

- **Multi-Model Comparison**: Tests 5 different models (BirdNET, Perch, VGG, MobileNet, ResNet)
- **K-Fold Cross-Validation**: Ensures robust performance evaluation with configurable folds
- **Embedding-Based Training**: Leverages pretrained models as feature extractors
- **Balanced Dataset Handling**: Automatically balances positive/negative samples
- **Configurable Experiments**: YAML-based configuration for flexible experimentation
- **Reproducible Results**: Deterministic sampling and model initialization

## Architecture

```
Audio Files → Feature Extraction → Custom Classifier Training → Performance Evaluation
     ↓              ↓                        ↓                         ↓
  data/*.wav    Embeddings/Features    Neural Networks          ROC-AUC Scores
```

## Directory Structure

```
src2/
├── README.md                    # This comprehensive guide
├── config.py                    # Configuration management
├── config.yaml                  # Main experiment configuration
├── test_config.yaml            # Test/development configuration
├── birdnet_perch_config.yaml   # BirdNET + Perch specific config
├── data_loader.py              # Audio loading and preprocessing
├── model_loader.py             # Model initialization and loading
├── trainer.py                  # Training and evaluation logic
├── main.py                     # Main experiment orchestration
├── results.py                  # Results saving and management
├── test_*.py                   # Test scripts for validation
└── PERCH_INTEGRATION.md        # Perch model integration guide
```

## File-by-File Documentation

### 1. `config.py` - Configuration Management

**Purpose**: Handles YAML configuration loading and validation.

**Key Classes**:
- `Config`: Data class containing experiment parameters
  - `models`: List of model names to test
  - `species`: List of species to classify
  - `training_sizes`: Sample sizes for training
  - `n_folds`: Number of K-fold cross-validation folds
  - `random_seed`: Seed for reproducible sampling
  - `data_path`: Path to audio data directory
  - `results_path`: Output directory for results

**Key Functions**:
- `load_config(config_path)`: Loads and parses YAML configuration

**Usage**:
```python
from config import load_config
config = load_config('config.yaml')
print(f"Models: {config.models}")
```

### 2. `data_loader.py` - Audio Data Pipeline

**Purpose**: Handles audio file loading, preprocessing, and K-fold split generation.

**Key Functions**:

#### `load_audio_files(data_path, species, training_size, datatype="data")`
- **Purpose**: Loads balanced audio datasets for training
- **Parameters**:
  - `data_path`: Base path to data directory
  - `species`: Species name (e.g., "coyote", "bullfrog")
  - `training_size`: Number of samples per class
  - `datatype`: "data" for regular files, "data_5s" for Perch model
- **Returns**: Tuple of (file_paths, labels)
- **Behavior**: 
  - Samples equal numbers of positive/negative files
  - Ensures balanced datasets for binary classification
  - Shuffles data for randomization

#### `preprocess_audio(file_path)`
- **Purpose**: Converts audio to mel spectrograms for PyTorch models
- **Process**:
  1. Loads audio with torchaudio
  2. Resamples to 16kHz if needed
  3. Converts to mel spectrogram (64 mel bins)
  4. Applies amplitude-to-dB transformation
  5. Pads/truncates to 128 time frames
- **Returns**: Tensor of shape (1, 64, 128)

#### `create_kfold_splits(files, labels, n_folds=5, seed=1)`
- **Purpose**: Creates stratified K-fold splits maintaining class balance
- **Returns**: List of (train_indices, validation_indices) tuples

### 3. `model_loader.py` - Model Architecture and Loading

**Purpose**: Initializes and manages all model types with unified interface.

#### Model Specifications

##### **BirdNET Model**
- **Type**: Embedding-based feature extractor
- **Architecture**: TensorFlow Lite model from BirdNET team
- **Input**: Raw audio files (any length)
- **Embedding Dimension**: 1024
- **Sample Rate**: 48kHz (internally handled)
- **Window Size**: 3 seconds
- **Data Source**: Regular `data/` folders
- **Use Case**: Bird vocalization analysis and transfer learning

```python
# BirdNET Usage
birdnet = BirdNETModel(model_path)
embeddings = birdnet.extract_embeddings(audio_file_path)  # Shape: (1, 1024)
```

##### **Perch Model**
- **Type**: Embedding-based feature extractor
- **Architecture**: EfficientNet backbone with PCEN frontend
- **Input**: 5-second audio segments
- **Embedding Dimension**: 1280
- **Sample Rate**: 32kHz
- **Window Size**: 5.0 seconds
- **Data Source**: `data_5s/` folders (5-second clips)
- **Use Case**: Bioacoustic classification and transfer learning
- **Model Source**: Google Research Perch (Kaggle Models)

```python
# Perch Usage
perch = PerchModel()
embeddings = perch.extract_embeddings(audio_file_path)  # Shape: (1, 1280)
```

##### **VGG Model**
- **Type**: End-to-end convolutional neural network
- **Architecture**: VGG-11 with modifications for audio
- **Input**: Mel spectrograms (1, 64, 128)
- **Modifications**:
  - First conv layer: 1 input channel (mono audio)
  - Final layer: 2 classes (binary classification)
- **Training**: Full end-to-end backpropagation

##### **ResNet Model**
- **Type**: End-to-end residual neural network
- **Architecture**: ResNet-18 adapted for audio
- **Input**: Mel spectrograms (1, 64, 128)
- **Modifications**:
  - First conv layer: 1 input channel
  - Final layer: 2 classes
- **Training**: Full end-to-end backpropagation

##### **MobileNet Model**
- **Type**: End-to-end efficient convolutional network
- **Architecture**: MobileNet-V2 for mobile/edge deployment
- **Input**: Mel spectrograms (1, 64, 128)
- **Modifications**:
  - First conv layer: 1 input channel
  - Final layer: 2 classes
- **Training**: Full end-to-end backpropagation

#### `load_model(model_name)`
- **Purpose**: Factory function for model instantiation
- **Returns**: Tuple of (model, device)
- **Supported Models**: "birdnet", "perch", "vgg", "resnet", "mobilenet"

### 4. `trainer.py` - Training and Evaluation Engine

**Purpose**: Handles model training and performance evaluation with different training strategies.

#### Training Strategies

##### **Embedding-Based Training** (BirdNET, Perch)
1. **Feature Extraction**: Extract embeddings from pretrained models
2. **Classifier Architecture**:
   ```
   Input (1024/1280) → Linear(256) → ReLU → Dropout(0.3) 
   → Linear(64) → ReLU → Dropout(0.3) → Linear(2)
   ```
3. **Training**: 20 epochs with Adam optimizer (lr=0.001)
4. **Benefits**: Fast training, leverages pretrained knowledge

##### **End-to-End Training** (VGG, ResNet, MobileNet)
1. **Direct Training**: Train entire network on mel spectrograms
2. **Training**: 5 epochs with Adam optimizer (lr=0.001)
3. **Benefits**: Task-specific feature learning

#### Key Functions

##### `train_model(model, train_files, train_labels, device, is_embedding_model=False)`
- **Purpose**: Trains models using appropriate strategy
- **Process**:
  - Embedding models: Extract features → Train classifier
  - End-to-end models: Train full network on spectrograms
- **Returns**: Trained model/classifier

##### `evaluate_model(model, test_files, test_labels, device, original_embedding_model=None)`
- **Purpose**: Evaluates model performance
- **Metric**: ROC-AUC score
- **Process**:
  - Generates predictions on test set
  - Computes area under ROC curve
- **Returns**: AUC score (float)

### 5. `main.py` - Experiment Orchestration

**Purpose**: Main execution script that orchestrates complete experiments.

#### Experiment Flow
1. **Configuration Loading**: Parse YAML configuration
2. **Species Iteration**: Loop through each species
3. **Model Iteration**: Test each model type
4. **Training Size Iteration**: Vary training data amounts
5. **K-Fold Cross-Validation**: Multiple train/test splits
6. **Result Aggregation**: Collect and save performance metrics

#### Key Functions

##### `main()`
- **Purpose**: Complete experiment execution
- **Process**:
  ```python
  for species in config.species:
      for model_name in config.models:
          for training_size in config.training_sizes:
              # Load balanced dataset
              files, labels = load_audio_files(...)
              
              # K-fold cross-validation
              for fold, (train_idx, val_idx) in enumerate(splits):
                  # Train model
                  trained_model = train_model(...)
                  
                  # Evaluate performance
                  auc_score = evaluate_model(...)
                  
              # Save results
              save_results(results, output_file)
  ```

**Usage**:
```bash
cd src2
python main.py  # Uses config.yaml by default
```

### 6. `results.py` - Results Management

**Purpose**: Handles saving and formatting of experiment results.

#### `save_results(results, output_path)`
- **Purpose**: Saves experiment results to CSV format
- **Output Format**:
  ```csv
  species,model,training_size,mean_auc,fold_scores
  coyote,birdnet,10,0.8234,"[0.8012, 0.8234, 0.8456]"
  coyote,perch,10,0.7891,"[0.7654, 0.7891, 0.8128]"
  ```

## Configuration Guide

### Main Configuration (`config.yaml`)

```yaml
# Logging and output settings
log_level: INFO
log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
save_plots: true
plot_format: png
plot_dpi: 300
max_workers: 4

experiments:
- name: full_experiment
  models:
  - vgg          # End-to-end CNN
  - resnet       # End-to-end ResNet
  - mobilenet    # End-to-end MobileNet
  - birdnet      # Embedding-based (uses data/)
  - perch        # Embedding-based (uses data_5s/)
  
  species:
  - coyote       # Available species
  - bullfrog
  # Add more species as needed
  
  training_sizes:
  - 10           # Small dataset
  - 25           # Medium dataset
  - 50           # Large dataset
  
  n_folds: 3           # K-fold cross-validation
  batch_size: 32       # Training batch size
  random_seed: 42      # Reproducibility
  kfold_seed: 1        # K-fold reproducibility
  
  # Paths
  data_path: /workspaces/non-avian-ml/data
  results_path: /workspaces/non-avian-ml/results
  datatype: data       # Overridden for perch model
```

### Data Directory Structure

```
data/
├── coyote/
│   ├── data/           # Regular audio files (for most models)
│   │   ├── pos/        # Positive samples (coyote vocalizations)
│   │   │   ├── file1.wav
│   │   │   └── file2.wav
│   │   └── neg/        # Negative samples (non-coyote sounds)
│   │       ├── file3.wav
│   │       └── file4.wav
│   └── data_5s/        # 5-second clips (for Perch model)
│       ├── pos/
│       └── neg/
├── bullfrog/
│   ├── data/
│   └── data_5s/
└── [other_species]/
    ├── data/
    └── data_5s/
```

## Usage Guide

### Quick Start

1. **Set up environment**:
   ```bash
   cd /workspaces/non-avian-ml/src2
   ```

2. **Configure experiment** (edit `config.yaml`):
   ```yaml
   models: [birdnet, perch]  # Start with embedding models
   species: [coyote]         # Single species test
   training_sizes: [10]      # Small dataset
   n_folds: 2               # Quick validation
   ```

3. **Run experiment**:
   ```bash
   python main.py
   ```

4. **Check results**:
   ```bash
   ls ../results/  # Look for CSV files
   ```

### Advanced Usage

#### Custom Configuration

Create a new configuration file:
```yaml
# my_experiment.yaml
experiments:
- name: my_custom_experiment
  models: [birdnet, vgg]
  species: [bullfrog]
  training_sizes: [5, 15, 30]
  n_folds: 5
  # ... other settings
```

Use custom configuration:
```python
from config import load_config
config = load_config('my_experiment.yaml')
# Then run main() with this config
```

#### Testing Individual Components

Test data loading:
```python
from data_loader import load_audio_files
files, labels = load_audio_files("/path/to/data", "coyote", 10)
print(f"Loaded {len(files)} files with {len(labels)} labels")
```

Test model loading:
```python
from model_loader import load_model
birdnet, device = load_model("birdnet")
print(f"Model loaded on {device}")
```

Test embedding extraction:
```python
embeddings = birdnet.extract_embeddings("path/to/audio.wav")
print(f"Embedding shape: {embeddings.shape}")
```

## Performance Expectations

### Training Times (Approximate)
- **BirdNET**: ~30 seconds per fold (embedding extraction + classifier training)
- **Perch**: ~30 seconds per fold (embedding extraction + classifier training)
- **VGG/ResNet/MobileNet**: ~2-5 minutes per fold (end-to-end training)

### Memory Requirements
- **Embedding Models**: Low GPU memory (~1-2GB)
- **End-to-End Models**: Higher GPU memory (~4-8GB)
- **Large Datasets**: More RAM for audio loading

### Expected Performance
- **BirdNET/Perch**: Often achieve AUC > 0.8 due to pretrained features
- **End-to-End Models**: Performance varies based on training data size
- **Small Datasets**: Embedding models typically outperform end-to-end models

## Troubleshooting

### Common Issues

#### 1. "No audio files found"
- **Cause**: Incorrect data path or missing audio files
- **Solution**: Check `data_path` in config and verify directory structure

#### 2. "CUDA out of memory"
- **Cause**: GPU memory insufficient for large models/batches
- **Solution**: Reduce `batch_size` in configuration

#### 3. "Model loading failed"
- **Cause**: Missing model files or dependencies
- **Solution**: Check BirdNET model file exists, verify Python environment

#### 4. "Insufficient samples for training"
- **Cause**: Not enough positive/negative samples for requested training size
- **Solution**: Reduce `training_sizes` or add more audio data

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Test individual components:
```bash
python test_birdnet_classifier.py    # Test BirdNET
python test_perch_classifier.py      # Test Perch
python test_birdnet_pipeline.py      # Test pipeline
```

## Development and Extension

### Adding New Models

1. **Create model class** in `model_loader.py`:
   ```python
   class MyCustomModel:
       def __init__(self):
           # Initialize model
           pass
       
       def extract_embeddings(self, audio_file_path):
           # Return embeddings
           pass
   ```

2. **Add to load_model function**:
   ```python
   elif model_name == "my_model":
       return MyCustomModel(), device
   ```

3. **Update configuration**:
   ```yaml
   models: [birdnet, perch, my_model]
   ```

### Adding New Species

1. **Create data directory**:
   ```
   data/new_species/
   ├── data/pos/
   ├── data/neg/
   ├── data_5s/pos/
   └── data_5s/neg/
   ```

2. **Update configuration**:
   ```yaml
   species: [coyote, bullfrog, new_species]
   ```

### Custom Evaluation Metrics

Extend `trainer.py`:
```python
from sklearn.metrics import precision_score, recall_score

def evaluate_model_extended(model, test_files, test_labels, device):
    # ... existing code ...
    
    # Add custom metrics
    precision = precision_score(test_labels, predictions > 0.5)
    recall = recall_score(test_labels, predictions > 0.5)
    
    return {
        'auc': roc_auc_score(test_labels, predictions),
        'precision': precision,
        'recall': recall
    }
```

## Model Integration Status

### Current Status
- **BirdNET**: Mock implementation (ready for TensorFlow Lite integration)
- **Perch**: Mock implementation (ready for Kaggle Models integration)
- **VGG/ResNet/MobileNet**: Fully functional PyTorch implementations

### Production Integration
- See `PERCH_INTEGRATION.md` for detailed Perch integration guide
- BirdNET integration requires TensorFlow Lite model loading
- All mock implementations provide consistent interfaces for seamless swapping

## Results Analysis

### Output Format
Results are saved to CSV files with columns:
- `species`: Species name
- `model`: Model name
- `training_size`: Number of training samples
- `mean_auc`: Average AUC across folds
- `fold_scores`: Individual fold AUC scores

### Visualization
Use the results for plotting:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/experiment_results.csv')

# Plot performance comparison
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    for model in species_data['model'].unique():
        model_data = species_data[species_data['model'] == model]
        plt.plot(model_data['training_size'], model_data['mean_auc'], 
                label=f'{model}', marker='o')
    
    plt.xlabel('Training Size')
    plt.ylabel('ROC-AUC')
    plt.title(f'Model Comparison - {species}')
    plt.legend()
    plt.show()
```

## License and Attribution

This framework is designed for research and educational purposes. Please cite appropriate sources when using pretrained models:

- **BirdNET**: BirdNET team (Cornell Lab of Ornithology)
- **Perch**: Google Research Perch team
- **PyTorch Models**: Meta/Facebook Research

## Support

For issues and questions:
1. Check this README for common solutions
2. Run test scripts to validate setup
3. Review configuration files for proper formatting
4. Check audio data directory structure

---

*This framework enables comprehensive comparison of embedding-based and end-to-end models for bioacoustic classification tasks, facilitating research in transfer learning and species-specific audio classification.*