# Non-Avian ML Audio Classification Experiment Framework (src2)

## Overview

This application is a modular machine learning experiment framework designed for comparing audio classification models across different species and training data sizes. The system supports both embedding-based models (BirdNET, Perch) and end-to-end CNN models (VGG, MobileNet, ResNet) for binary classification tasks.

## Key Features

- **Multi-Model Comparison**: Tests embedding models (BirdNET, Perch) and CNN models (VGG, MobileNet, ResNet)
- **Multiple Random Seeds**: Evaluates model robustness across different training data draws
- **K-Fold Cross-Validation**: Ensures robust performance evaluation with configurable folds
- **Embedding-Based Training**: Leverages pretrained models as feature extractors for faster training
- **Balanced Dataset Handling**: Automatically balances positive/negative samples for each random seed
- **Statistical Analysis**: Computes confidence intervals across random seed variations
- **Automated Visualization**: Generates performance plots with error bars showing data sensitivity
- **Configurable Experiments**: YAML-based configuration for flexible experimentation
- **Reproducible Results**: Deterministic sampling and model initialization

## Architecture

```
Audio Files → Model-Specific Processing → Training → Evaluation → Aggregation
     ↓              ↓                        ↓           ↓            ↓
  data/*.wav    BirdNET/Perch:          Custom      ROC-AUC    Mean ± Std
               Embeddings              Classifiers   Scores    Across Seeds
               
               CNN Models:
               Spectrograms
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) with the following structure:

```yaml
log_level: INFO
log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
save_plots: true
plot_format: png
plot_dpi: 300
max_workers: 4
experiments:
- name: all_models_test
  models: [birdnet, perch]  # Available: birdnet, perch, vgg, mobilenet, resnet
  species: [coyote, bullfrog]
  training_sizes: [10, 25]  # Number of samples per class
  n_folds: 3               # K-fold cross-validation folds
  batch_size: 32
  random_seeds: [1,2,3,4,5]  # Multiple data draws for robustness
  kfold_seed: 1            # Consistent K-fold splits across seeds
  data_path: /workspaces/non-avian-ml/data
  results_path: /workspaces/non-avian-ml/results
  datatype: data          # "data" or "data_5s" (auto-selected for Perch)
```

### Key Configuration Parameters

- **`random_seeds`**: List of seeds for different training data draws. Each seed creates a different random sample of training data, allowing measurement of model robustness to data selection.
- **`kfold_seed`**: Fixed seed for K-fold splitting to ensure consistent evaluation across random seeds.
- **`models`**: 
  - `birdnet`: Uses TensorFlow Lite model for 1024-dim embeddings
  - `perch`: Uses EfficientNet-based model for 1280-dim embeddings (uses data_5s)
  - `vgg`, `mobilenet`, `resnet`: End-to-end CNN training on spectrograms
- **`training_sizes`**: Number of positive and negative samples per training set
- **`datatype`**: Automatically set to "data_5s" for Perch model, "data" for others

## Usage

### Quick Start

1. **Run the main experiment:**
```bash
cd /workspaces/non-avian-ml
pixi run python src2/main.py
```

2. **View results:**
   - CSV: `/workspaces/non-avian-ml/results/experiment_results.csv`
   - Plot: `/workspaces/non-avian-ml/results/experiment_results_plot_species_models.png`

### Running Specific Configurations

```bash
# Use a different config file
pixi run python src2/main.py --config test_config.yaml

# Run from project root (auto-detects src2/config.yaml)
cd /workspaces/non-avian-ml
pixi run python src2/main.py
```

### Understanding Results

The system outputs:

1. **Aggregated CSV** with columns:
   - `model`: Model name (birdnet, perch, etc.)
   - `species`: Species name (coyote, bullfrog, etc.)
   - `training_size`: Number of samples per class
   - `mean_auc`: Average ROC-AUC across random seeds
   - `std_auc`: Standard deviation showing data sensitivity

2. **Visualization Plot** showing:
   - X-axis: Training size
   - Y-axis: ROC-AUC performance
   - Error bars: ±1 standard deviation across random seeds
   - Separate subplots for each species
   - Different lines for each model

### Statistical Interpretation

- **Mean AUC**: Average performance across different training data samples
- **Standard Deviation**: Indicates model robustness to training data selection
  - Low std_auc = Model is robust to different training samples
  - High std_auc = Model performance varies significantly with data selection
- **Error Bars**: Show confidence intervals for expected performance range

## Data Structure

Expected data organization:
```
data/
├── species_name/
│   ├── data/           # For BirdNET, VGG, MobileNet, ResNet
│   │   ├── pos/        # Positive samples
│   │   └── neg/        # Negative samples
│   └── data_5s/        # For Perch model (5-second segments)
│       ├── pos/
│       └── neg/
```

## Models

- **BirdNET**: TensorFlow Lite, 1024-dim embeddings, 3s windows
- **Perch**: EfficientNet-based, 1280-dim embeddings, 5s windows
- **VGG/MobileNet/ResNet**: End-to-end CNN training on spectrograms

## Development



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
- **BirdNET**: Real TensorFlow Lite integration using ai-edge-litert. Extracts
   the true 1024-d embedding (GLOBAL_AVG_POOL/Mean) from the official TFLite
   model you have in the repository.
- **Perch**: Real integration path via google-research/perch (chirp). Requires
   installing `chirp` from the GitHub repo; see `PERCH_INTEGRATION.md`.
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