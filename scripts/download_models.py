"""
download_models.py

Every model downloads itself on first use into models/ (gitignored) and is reused after that, so this
script is optional: it fetches all six up front (e.g. before running on a server or offline).

  BirdNET 2.4 (TFLite FP32), BirdNET 3.0, Perch 2.0  -> models/birdnet_cache/   (birdnet package)
  Perch v8                                          -> models/perch_v8/         (Kaggle)
  PANNs CNN14 + AudioSet labels                     -> models/cnn14/            (Zenodo)
  ResNet-18 ImageNet weights                        -> models/torch/            (torchvision)

Usage:
    pixi run python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import model_files  # noqa: E402  (sets BIRDNET_APP_DATA / TORCH_HOME first)


def main():
    path, labels = model_files.birdnet_v2_4()
    print(f"BirdNET 2.4: {path} ({len(labels)} labels)")
    path, labels = model_files.birdnet_v3_0()
    print(f"BirdNET 3.0: {path} ({len(labels)} labels)")
    print(f"Perch v8:    {model_files.perch_v8()}")
    path, labels = model_files.perch_v2()
    print(f"Perch 2.0:   {path} ({len(labels)} labels)")
    path, labels = model_files.cnn14()
    print(f"CNN14:       {path} ({len(labels)} labels)")
    from torchvision.models import ResNet18_Weights, resnet18
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet-18:   torchvision IMAGENET1K_V1 weights cached in models/torch")
    print("Done.")


if __name__ == "__main__":
    main()
