# Perch Model Integration Notes

## Current Implementation
- **Status**: Real integration via google-research/perch (chirp). Requires installing
  the `chirp` inference package from the Perch repository.
- **Features**:
  - 5-second audio windows at 32 kHz (data_5s folders)
  - Official EfficientNet Perch embeddings (1280-d) via chirp.inference
  - L2-normalized embeddings for downstream classifiers

## Production Integration Path

To integrate the actual Perch model from Google Research:

### Option 1: Install chirp from GitHub (recommended)

Add this dependency (already added in pixi.toml):

```
chirp = @ git+https://github.com/google-research/perch.git
```

Then rebuild your environment. For pixi:

```
pixi install
```

Verify installation:

```
pixi run python -c "import chirp; print('chirp OK')"
```

### Option 2: Using Perch repository with Poetry

Follow the official README for Poetry installation:

```
git clone https://github.com/google-research/perch
cd perch
poetry install
poetry run python -m unittest discover -s chirp/inference/tests -p "*test.py"
```

### Notes on TensorFlow SavedModels
Perch models are exported to TensorFlow; some integration paths require
`tensorflow` availability. If TF import fails on your CPU, prefer using the
Perch Docker or a machine with compatible CPU flags.

## Integration Requirements
1. Audio: 5 s segments at 32 kHz
2. Frontend: PCEN melspectrogram (handled by chirp)
3. Model: EfficientNet backbone (Perch)
4. Output: 1280-d embeddings
5. Normalization: L2 normalized

## Validation
- Test script (after installing chirp):

```
python - << 'PY'
from src2.model_loader import PerchModel
m = PerchModel()
wav = 'data/bullfrog/data_5s/neg/yourfile.wav'
emb = m.extract_embeddings(wav)
print('Perch embedding:', emb.shape, emb.norm().item())
PY
```

## Next Steps
1. Ensure chirp is installed (pixi install or Poetry route)
2. Run a quick embedding check as above
3. Train classifiers on the 1280-d embeddings via the existing pipeline