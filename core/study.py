"""What every experiment starts from: a clip manifest, its train/test split by source recording, each target
class's binary split, and cached embeddings as a matrix on the training device.

All experiments except window length use `load_study(cfg)`, so each class has the same test clips and the same
training draws in every experiment.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.data import Partition, Split, binary_split, load_manifest, make_partition
from core.embeddings import EmbeddingTable, get_embeddings
from core.encoders import torch_device


@dataclass
class Study:
    manifest: pd.DataFrame
    partition: Partition
    splits: dict[str, Split]
    data_path: Path | None = None
    embeddings_path: Path | None = None
    clip_seconds: float | None = None  # input window, if not the default 3 s

    def __post_init__(self):
        self.cls_of = dict(zip(self.manifest.filename, self.manifest.cls))

    def embed(self, model: str, files) -> tuple[EmbeddingTable, torch.Tensor]:
        """Embeddings of these clips (encoding any not yet cached) and the embedding matrix on the training device."""
        clips = sorted({(self.cls_of[f], f) for f in files})
        table = get_embeddings(model, clips, self.data_path, self.embeddings_path, self.clip_seconds)
        return table, torch.from_numpy(np.ascontiguousarray(table.emb)).to(torch_device())


def make_study(manifest: pd.DataFrame, classes: list[str], cfg, test_size: int, min_train: int, **paths) -> Study:
    partition = make_partition(manifest, classes, test_size, cfg.split_seed, cfg.max_test_recording_clips,
                               min_train, cfg.donor_test_fraction, cfg.low_sample_test_fraction)
    return Study(manifest, partition, {c: binary_split(partition, c, cfg.split_seed) for c in classes}, **paths)


def load_study(cfg) -> Study:
    """The main clip set (data/metadata.csv) and the shared split."""
    return make_study(load_manifest(cfg.clips_csv), cfg.species, cfg, cfg.test_size_per_class, cfg.min_train_clips,
                      data_path=cfg.data_path, embeddings_path=cfg.embeddings_path)
