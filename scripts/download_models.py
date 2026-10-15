"""Every model downloads itself on first use into cache/models/ (gitignored) and is reused after that, so this
script is optional: it fetches all six up front (e.g. before running on a server or offline).

  BirdNET 2.4 (TFLite FP32), BirdNET 3.0, Perch 2.0  -> cache/models/birdnet_cache/   (birdnet package)
  Perch v8                                          -> cache/models/perch_v8/         (Kaggle)
  PANNs CNN14 + AudioSet labels                     -> cache/models/cnn14/            (Zenodo)
  ResNet-18 ImageNet weights                        -> cache/models/torch/            (torchvision)

Usage:
    pixi run download-models
"""
from core import encoders


def main():
    path, labels = encoders.birdnet_v2_4()
    print(f"BirdNET 2.4: {path} ({len(labels)} labels)")
    path, labels = encoders.birdnet_v3_0()
    print(f"BirdNET 3.0: {path} ({len(labels)} labels)")
    print(f"Perch v8:    {encoders.perch_v8()}")
    path, labels = encoders.perch_v2()
    print(f"Perch 2.0:   {path} ({len(labels)} labels)")
    path, labels = encoders.cnn14()
    print(f"CNN14:       {path} ({len(labels)} labels)")
    from torchvision.models import ResNet18_Weights, resnet18
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print("ResNet-18:   torchvision IMAGENET1K_V1 weights cached in cache/models/torch")
    print("Done.")


if __name__ == "__main__":
    main()
