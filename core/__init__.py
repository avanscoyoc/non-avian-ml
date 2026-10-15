"""Shared machinery for the experiments in experiments/; nothing here runs an experiment by itself.

  data        clip manifest, the train/test split by source recording, binary splits and training draws
  encoders    the six frozen pretrained models, their model files and audio loading
  embeddings  per-model embedding cache
  study       what every experiment starts from: manifest + split + embeddings on the training device
  heads       MLP classifier heads, many trained at once
  binary      binary detector learning curves (saturation and window-length experiments)
  zero_shot   scoring clips with a model's own labels
  plotting    shared figure settings and helpers
"""
from core import paths  # noqa: F401  (sets BIRDNET_APP_DATA / TORCH_HOME before any model library is imported)
