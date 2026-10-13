"""Shared plot settings: model colours/markers (same in every figure), chart chrome and class display names.

Model colours are from seaborn's "deep" palette, grouped by family: BirdNET blue/purple, Perch red/orange,
CNN14 green, ResNet brown. Each model always keeps its colour. Lines can cross, so every model also has its
own marker shape; identity never rests on colour alone, and figures always carry a legend.
"""
import matplotlib.cm as mplcm

MODELS = {  # plotting order -> (legend name, colour, marker)
    "birdnet": ("BirdNET 2.4", "#4c72b0", "o"),
    "birdnet3": ("BirdNET 3.0", "#8172b3", "s"),
    "perch": ("Perch v8", "#c44e52", "^"),
    "perch2": ("Perch 2.0", "#dd8452", "D"),
    "cnn14": ("CNN14 (AudioSet)", "#55a868", "v"),
    "resnet": ("ResNet-18 (ImageNet)", "#937860", "P"),
}

# Chart chrome (light surface)
INK = "#0b0b0b"          # titles, primary text
INK_SECONDARY = "#52514e"  # tick labels, secondary text (e.g. "low sample")
INK_MUTED = "#898781"    # reference lines
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # spines / baseline

CLASS_COLORS = [mplcm.tab10(i) for i in range(10)] + [mplcm.tab20b(i) for i in range(0, 20, 4)]


def model_name(model: str) -> str:
    return MODELS.get(model, (model, None, None))[0]


def model_color(model: str) -> str:
    return MODELS.get(model, (None, "#52514e", None))[1]


def model_marker(model: str) -> str:
    return MODELS.get(model, (None, None, "o"))[2]


def class_name(cls: str) -> str:
    return cls.replace("_", " ").capitalize()
