# Non-Avian ML Audio Classification

## Overview

A machine learning framework for comparing audio classification models across different training set sizes. Evaluates performance using frozen pretrained embeddings from five models on binary classification tasks (e.g., coyote presence/absence).

**Workflow:** Extract frozen embeddings → Train MLP classifier head → Evaluate on fixed test set

## Models

| Model | Embedding Dim | Pretrained On | Audio Input |
|-------|--------------|---------------|-------------|
| **BirdNET** | 6522-D (logits) | Bird species (TFLite) | 3s @ 48kHz |
| **Perch** | 1280-D | Bird vocalizations | 5s @ 32kHz |
| **ResNet18** | 512-D | ImageNet | Mel-spectrogram |
| **MobileNetV2** | 1280-D | ImageNet | Mel-spectrogram |
| **VGG11** | 4096-D | ImageNet | Mel-spectrogram |

## Usage

### Configuration

Edit `src/config.yaml`:

```yaml
experiments:
- name: all_models_test
  models: [birdnet, perch, resnet, mobilenet, vgg]
  species: [coyote, bullfrog]
  training_sizes: [10, 20, 40, 60, 80, 100, 120]  # Samples per class
  n_folds: 5                              # K-fold CV folds
  n_epochs: 20                            # Training epochs 
  test_size_per_class: 50                 # Fixed test set size
  batch_size: 32                          # Batch size for training
  random_seeds: [1,2,3,4,5,6,7,8,9,10]   # Multiple runs for statistics
  kfold_seed: 1                           # Fixed for reproducibility
  data_path: /workspaces/non-avian-ml/data
  results_path: /workspaces/non-avian-ml/results
  datatype: data                          # "data" or "data_5s"
```

**Key Parameters:**
- `training_sizes`: Sample sizes per class to evaluate (creates learning curves)
- `n_folds`: Number of cross-validation folds (5 is standard)
- `n_epochs`: Training epochs for MLP classifier (20 typical for convergence)
- `batch_size`: Training batch size (32 is standard, reduce if memory limited)
- `random_seeds`: Multiple runs with different training samples (10 recommended for publication)

### Run Experiment

```bash
cd /workspaces/non-avian-ml
pixi run python src/main.py
```

### Results

![comparison_curves](/workspaces/non-avian-ml/figs/species_comparison.png)
Fig 1. Current performance of species classes by training size for 5 model architectures. 

**CSV Output (`experiment_results.csv`):**

| Column | Description |
|--------|-------------|
| `model`, `species`, `training_size` | Experiment parameters |
| `n_folds`, `n_epochs`, `batch_size`, `test_size_per_class` | Configuration used |
| `cv_auc_mean`, `cv_auc_std`, `cv_auc_ci_95`, `cv_auc_n` | Cross-validation statistics |
| `test_auc_mean`, `test_auc_std`, `test_auc_ci_95`, `test_auc_n` | Test set statistics |

- **`*_mean`**: Average performance across random seeds
- **`*_std`**: Standard deviation (measures variability)
- **`*_ci_95`**: 95% confidence interval (±CI around mean)
- **`*_n`**: Number of runs aggregated

**Plot (`experiment_results_plot.png`):**
- Learning curves showing test AUC vs. training size
- Error bars represent 95% confidence intervals
- Separate subplot per species

# Current Performance
Class specific AUC-RUC is measured on the diagonal, showing high performance on all classes except Engine and Human vocal (currently due to low sample size). Limited confusion between frog species due to sampling. 

![Confusion Matrix](workspaces/non-avian-ml/figs/summary_heatmap_f1_score.png)

## Data Structure

```
data/
└── {species}/
    ├── data/              # 3s clips for BirdNET, CNNs
    │   ├── pos/*.wav
    │   └── neg/*.wav
    └── data_5s/           # 5s clips for Perch
        ├── pos/*.wav
        └── neg/*.wav
```

**Minimum files per class:** 150+ (50 test + 100 train + buffer)

## Key Features

- **Fixed test set:** Identical 100 samples (50 pos/neg) for all experiments
- **Frozen embeddings:** Pretrained models used as feature extractors
- **Stratified K-fold CV:** Robust within-training evaluation
- **Statistical rigor:** 95% confidence intervals from multiple runs
- **Fully reproducible:** Deterministic splits with fixed seeds

## Installation

```bash
# Install dependencies via pixi
pixi install

# Or manually install required packages:
# PyTorch, torchaudio, torchvision
# TensorFlow, ai-edge-litert, tensorflow-hub
# scikit-learn, librosa, soundfile
# pandas, matplotlib, numpy
```

## Citation

```bibtex
@software{non_avian_ml_2025,
  title = {Non-Avian ML Audio Classification Framework},
  author = {Van Scoyoc, Amy},
  year = {2025},
  url = {https://github.com/avanscoyoc/non-avian-ml}
}
```
}
```