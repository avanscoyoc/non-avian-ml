"""Repo-relative paths, so the code runs from any clone (devcontainer, cirrus, laptop)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS_CSV = DATA / "metadata.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

# Every model is downloaded on first use into models/ (gitignored); see src/model_files.py.
# Must be set before `birdnet` / `torch` are imported, since they read these at import time.
os.environ.setdefault("BIRDNET_APP_DATA", str(MODELS / "birdnet_cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))


def resolve(path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
