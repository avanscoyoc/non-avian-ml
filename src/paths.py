"""Repo-relative paths, so the code runs from any clone (devcontainer, cirrus, laptop)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS_CSV = DATA / "metadata.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

# Downloaded models (BirdNET 3.0, Perch 2.0, CNN14) live under models/ (gitignored).
# Must be set before `birdnet` is imported, since it reads this at import time.
os.environ.setdefault("BIRDNET_APP_DATA", str(MODELS / "birdnet_cache"))


def resolve(path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
