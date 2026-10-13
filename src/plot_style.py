"""Shared plot settings: model names/colours (same in every figure) and class display names."""
import matplotlib.cm as mplcm

MODELS = {  # plotting order -> (legend name, colour)
    "birdnet": ("BirdNET 2.4", "#2E86AB"),
    "birdnet3": ("BirdNET 3.0", "#1B4F72"),
    "perch": ("Perch v8", "#F18F01"),
    "perch2": ("Perch 2.0", "#C73E1D"),
    "cnn14": ("CNN14 (AudioSet)", "#A23B72"),
    "resnet": ("ResNet-18 (ImageNet)", "#06A77D"),
}

CLASS_COLORS = [mplcm.tab10(i) for i in range(10)] + [mplcm.tab20b(i) for i in range(0, 20, 4)]


def model_name(model: str) -> str:
    return MODELS.get(model, (model, None))[0]


def model_color(model: str) -> str:
    return MODELS.get(model, (None, "#333333"))[1]


def class_name(cls: str) -> str:
    return cls.replace("_", " ").capitalize()
