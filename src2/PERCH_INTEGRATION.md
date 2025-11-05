# Perch Model Integration Notes

## Current Implementation
- **Status**: Mock implementation for development/testing
- **Purpose**: Demonstrates the embedding extraction workflow for custom classifier training
- **Features**: 
  - Correct 5-second audio window handling (data_5s folders)
  - Proper embedding dimensions (1280, matching EfficientNet)
  - Deterministic embeddings for reproducibility

## Production Integration Path

To integrate the actual Perch model from Google Research:

### Option 1: Using Kaggle Models API
```python
# Install kaggle models
# pip install kaggle-models

from kaggle_models import BirdVocalizationClassifier
model = BirdVocalizationClassifier.from_pretrained()

# Extract embeddings
embeddings = model.embed(audio_data)  # 5-second audio at 32kHz
```

### Option 2: Using Perch Repository Directly
```python
# Clone and install perch repository
# git clone https://github.com/google-research/perch
# pip install ./perch

from chirp.inference import embed_lib
from chirp.inference import zoo_interface

# Load Perch model
model = zoo_interface.get_model('perch_8')
outputs = model.embed(audio_data)
embeddings = outputs.embeddings
```

### Option 3: Using TensorFlow Hub/SavedModel
```python
import tensorflow_hub as hub

# Load Perch from TensorFlow Hub (if available)
model = hub.load("https://tfhub.dev/google/perch/1")
embeddings = model(audio_tensor)
```

## Integration Requirements
1. **Audio Format**: 5-second segments at 32kHz sample rate
2. **Frontend**: PCEN melspectrogram preprocessing
3. **Model**: EfficientNet-based backbone
4. **Output**: 1280-dimensional embeddings
5. **Normalization**: L2 normalized embeddings

## Current Mock Behavior
- Uses deterministic hash-based embeddings for consistent results
- Different seed space than BirdNET to ensure feature diversity
- Maintains same interface as real model for seamless swapping
- Supports all pipeline operations (training, evaluation, K-fold CV)

## Next Steps for Production
1. Choose integration method (Kaggle, Perch repo, or TensorFlow Hub)
2. Replace `PerchModel.__init__()` and `extract_embeddings()` methods
3. Add error handling for model loading
4. Test with actual audio data
5. Validate embedding quality and classifier performance