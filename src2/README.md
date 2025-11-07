# Non-Avian ML Audio Classification - Training Set Size Comparison (src2)

## Overview

`src2` is a machine learning experiment framework for comparing audio classification models across different training set sizes. It evaluates performance using **frozen pretrained embeddings** from five models on binary classification tasks (e.g., coyote presence/absence).

**Key Design:** All models use the same workflow:
1. Extract frozen pretrained embeddings from audio
2. Train a small MLP classifier head on embeddings
3. Evaluate on a fixed held-out test set

This ensures fair comparison across models and training sizes.

## Models

All models use **frozen pretrained weights** as feature extractors:

| Model | Embedding Dim | Pretrained On | Audio Input |
|-------|--------------|---------------|-------------|
| **BirdNET** | 6522-D (logits) | Bird species (TFLite) | 3s @ 48kHz |
| **Perch** | 1280-D | Bird vocalizations | 5s @ 32kHz |
| **ResNet18** | 512-D | ImageNet | Mel-spectrogram |
| **MobileNetV2** | 1280-D | ImageNet | Mel-spectrogram |
| **VGG11** | 4096-D | ImageNet | Mel-spectrogram |

## Key Features

- ✅ **Fixed test set**: Same 100-sample test set (50 pos + 50 neg) for all experiments
- ✅ **Frozen embeddings**: No fine-tuning of pretrained models
- ✅ **Stratified K-fold CV**: Robust training within each sample size
- ✅ **Multiple random seeds**: Measures sensitivity to training data selection
- ✅ **Deterministic splits**: Fully reproducible results
- ✅ **Fair comparison**: All models use same train/test split per species

## Directory Structure

```
src2/
├── main.py              # Main experiment runner
├── config.yaml          # Experiment configuration
├── config.py            # Config loader
├── data_loader.py       # Train/test split & audio loading
├── model_loader.py      # Model initialization (frozen extractors)
├── trainer.py           # Classifier training on embeddings
├── results.py           # Results aggregation & plotting
└── README.md            # This file
```

## Configuration

Edit `src2/config.yaml`:

```yaml
experiments:
- name: all_models_test
  models: [birdnet, perch, resnet, mobilenet, vgg]
  species: [coyote, bullfrog, ...]
  training_sizes: [10, 25, 50, 75, 100]  # Samples per class
  n_folds: 5                              # K-fold CV folds
  test_size_per_class: 50                 # Fixed test set size
  random_seeds: [1,2,3,4,5,6,7,8,9,10]   # Training data draws
  kfold_seed: 1                           # Fixed for consistent CV splits
  data_path: /workspaces/non-avian-ml/data
  results_path: /workspaces/non-avian-ml/results
  datatype: data                          # "data" or "data_5s"
```

### Key Parameters

- **`test_size_per_class`**: Fixed number of test samples per class (default: 50)
  - Ensures fair cross-species comparison
  - Same test set for all training sizes
  
- **`training_sizes`**: List of training samples per class to test
  - Creates learning curves showing performance vs. data size
  
- **`random_seeds`**: Multiple training data draws for robustness
  - Each seed samples different training files from train pool
  - Results aggregated across seeds (mean ± std)
  
- **`kfold_seed`**: Fixed seed for K-fold splitting
  - Ensures consistent validation splits across experiments

## Usage

### Run Full Experiment

```bash
cd /workspaces/non-avian-ml
pixi run python src2/main.py
```

### Output Files

```
results/
├── experiment_results.csv          # Aggregated results
└── experiment_results_plot.png     # Learning curves
```

### Results Format

**CSV columns:**
- `model`: Model name
- `species`: Species name  
- `training_size`: Training samples per class
- `cv_auc_mean`: Mean cross-validation AUC (for model selection)
- `cv_auc_std`: Std dev of CV AUC across folds
- `test_auc_mean`: **Mean test AUC across random seeds** ← Use this for comparison
- `test_auc_std`: Std dev across random seeds (data sensitivity)

**Plot:** Learning curves showing `test_auc_mean` ± `test_auc_std` vs. training size.

## Workflow

### 1. Data Splitting (Once per Species)

```python
# Fixed 50 samples per class for test set
train_pos, train_neg, test_pos, test_neg = create_train_test_split(
    data_path, species, datatype, test_size_per_class=50, seed=kfold_seed
)
# test_pos: 50 files, test_neg: 50 files (FIXED for all experiments)
```

### 2. Training Loop (Per training_size, per seed)

```python
for training_size in [10, 25, 50, 75, 100]:
    for seed in random_seeds:
        # Sample from train pool
        files, labels = load_audio_files(training_size, train_pos, train_neg, seed)
        
        # K-fold CV for robust training
        for train_files, val_files in create_kfold_splits(files, labels, n_folds):
            model = load_model(model_name)  # Frozen feature extractor
            classifier = train_model(model, train_files, labels, device)
            val_auc = evaluate_model(classifier, val_files, val_labels, device, model)
        
        # Final evaluation on fixed test set
        test_auc = evaluate_model(classifier, test_files, test_labels, device, model)
```

### 3. Classifier Training (Embedding-Based)

```python
# 1. Extract frozen embeddings
embeddings = [model.extract_embeddings(f) for f in train_files]  # Shape: [N, D]

# 2. Train small MLP head
classifier = nn.Sequential(
    nn.Linear(embedding_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2),  # Binary classification
)

# 3. Train for 20 epochs on embeddings
# Only the classifier head is trained; embeddings are frozen
```

## Data Requirements

### Directory Structure

```
data/
└── {species}/
    ├── data/              # 3s clips for BirdNET, CNNs
    │   ├── pos/
    │   │   └── *.wav
    │   └── neg/
    │       └── *.wav
    └── data_5s/           # 5s clips for Perch
        ├── pos/
        │   └── *.wav
        └── neg/
            └── *.wav
```

### Minimum Requirements

For `test_size_per_class=50` and `max(training_sizes)=100`:
- **Minimum per class**: 150 files (50 test + 100 train)
- **Recommended**: 200+ files per class for larger training sizes

## Key Implementation Details

### Frozen Embeddings

All models extract features with frozen weights:

```python
# BirdNET: 6522-D logits from TFLite model
birdnet_emb = model.extract_embeddings(audio_file)  # Shape: [1, 6522]

# Perch: 1280-D from EfficientNet
perch_emb = model.extract_embeddings(audio_file)    # Shape: [1, 1280]

# CNNs: Extract from penultimate layer
cnn_emb = model(mel_spectrogram)                    # Shape: [1, 512/1280/4096]
```

All embeddings are L2-normalized before classification.

### Deterministic Behavior

Results are fully reproducible:

1. **Test set**: Files sorted before shuffling → same test set for same `kfold_seed`
2. **Training samples**: Fixed `random_seed` → same training files
3. **K-fold splits**: Fixed `kfold_seed` → same validation folds
4. **Model initialization**: PyTorch/TensorFlow defaults are deterministic

### Test Set vs. Cross-Validation

- **Cross-validation AUC** (`cv_auc_mean`): Used during model development
- **Test set AUC** (`test_auc_mean`): Used for final comparison ✅

The test set is **never seen during training** and is **identical across all training sizes**, ensuring fair comparison.

## Troubleshooting

### BirdNET Not Learning

- ✅ **Fixed**: Now uses 6522-D logits instead of trying to extract intermediate tensors
- BirdNET may show ~98% AUC even at small sample sizes (pretrained features are very strong)

### Results Not Reproducible

- ✅ **Fixed**: Files are now sorted before shuffling to ensure deterministic splits
- Same seeds will produce identical results across runs

### Perch Import Error

If `import chirp` fails:
```bash
pixi install
pixi run pip install git+https://github.com/google-research/perch.git
```

### Out of Memory

Reduce:
- `training_sizes` (fewer/smaller sizes)
- `random_seeds` (fewer repetitions)
- Embedding models (exclude CNNs)

## Expected Performance

Typical results on well-separated tasks (e.g., coyote presence):

| Model | 10 samples | 25 samples | 50 samples | 100 samples |
|-------|-----------|-----------|-----------|------------|
| BirdNET | ~98% | ~99% | ~99% | ~99% |
| Perch | ~95% | ~97% | ~98% | ~99% |
| VGG11 | ~97% | ~99% | ~99% | ~99% |
| ResNet18 | ~65% | ~75% | ~85% | ~90% |
| MobileNetV2 | ~55% | ~60% | ~70% | ~80% |

**Note:** Performance depends heavily on:
- Task difficulty (how distinct the classes are)
- Audio quality
- Species-specific challenges

## Citation

If using this framework, cite:
```
@software{non_avian_ml_2025,
  title = {Non-Avian ML Audio Classification Framework},
  author = {Van Scoyoc, Amy},
  year = {2025},
  url = {https://github.com/avanscoyoc/non-avian-ml}
}
```

## Dependencies

Installed via `pixi.toml`:
- PyTorch, torchaudio (CNN models)
- TensorFlow, ai-edge-litert (BirdNET TFLite)
- chirp-inference (Perch embeddings)
- scikit-learn (metrics, K-fold)
- librosa, soundfile (audio processing)
- pandas, matplotlib, seaborn (results)
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