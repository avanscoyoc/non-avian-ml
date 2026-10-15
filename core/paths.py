"""Repo-relative paths, so the code runs from any clone (devcontainer, laptop, GPU server)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
DATA = ROOT / "data"
MODELS = ROOT / "cache" / "models"

# Every model is downloaded on first use into cache/models/ (gitignored); see core/encoders.py.
# Must be set before `birdnet` / `torch` are imported, since they read these at import time.
os.environ.setdefault("BIRDNET_APP_DATA", str(MODELS / "birdnet_cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))


def resolve(path) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path
