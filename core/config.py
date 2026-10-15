"""Settings shared by every experiment, read from config.yaml (one flat mapping, documented there)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.paths import CONFIG, resolve


@dataclass
class Config:
    models: list[str]
    classes: list[str]
    class_types: dict[str, list[str]]
    training_sizes: list[int]
    headline_n: int
    random_seeds: list[int]
    n_epochs: int
    batch_size: int
    split_seed: int
    test_size_per_class: int
    max_test_recording_clips: int
    min_train_clips: int
    low_sample_test_fraction: float
    negative_test_fraction: float
    data_path: Path
    clips_csv: Path
    embeddings_path: Path
    results_path: Path
    figs_path: Path
    zero_shot_labels: dict[str, dict[str, list[str]]]  # {class: {model: [label, ...]}}

    def __post_init__(self):
        for name in ("data_path", "clips_csv", "embeddings_path", "results_path", "figs_path"):
            setattr(self, name, resolve(getattr(self, name)))


def load_config(path=CONFIG) -> Config:
    with open(path) as f:
        return Config(**yaml.safe_load(f))
