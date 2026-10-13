import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from src.paths import CLIPS_CSV, DATA, RESULTS, resolve
from src.data_loader import Partition, make_partition



@dataclass
class Config:
    models: List[str]
    species: List[str]
    training_sizes: Optional[List[int]] = None
    n_epochs: int = 20
    test_size_per_class: int = 50
    batch_size: int = 32
    random_seeds: List[int] = None
    split_seed: int = 1
    max_test_recording_clips: int = 25
    min_train_clips: int = 40
    low_sample_test_fraction: float = 0.5
    donor_test_fraction: float = 0.2
    class_types: Dict[str, List[str]] = field(default_factory=dict)
    multiclass_groups: Dict[str, List[str]] = field(default_factory=dict)
    multiclass_train_size: int = 40
    data_path: Path = DATA
    clips_csv: Path = CLIPS_CSV
    results_path: Path = RESULTS
    embeddings_path: Path = RESULTS / "embeddings"
    save_classifier: bool = False
    species_sizes: Optional[Dict[str, Dict[str, int]]] = None
    run_zeroshot: bool = True


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    exp = data["experiments"][0]
    species_sizes = exp.get("species_sizes", None)
    species = exp.get("species", list(species_sizes.keys()) if species_sizes else [])
    data_path = resolve(exp.get("data_path", DATA))
    results_path = resolve(exp.get("results_path", RESULTS))
    return Config(
        models=exp["models"],
        species=species,
        training_sizes=exp.get("training_sizes", None),
        n_epochs=exp.get("n_epochs", 20),
        test_size_per_class=exp.get("test_size_per_class", 50),
        batch_size=exp.get("batch_size", 32),
        random_seeds=exp.get("random_seeds", [42]),
        split_seed=exp.get("split_seed", exp.get("kfold_seed", 1)),
        max_test_recording_clips=exp.get("max_test_recording_clips", 25),
        min_train_clips=exp.get("min_train_clips", 40),
        low_sample_test_fraction=exp.get("low_sample_test_fraction", 0.5),
        donor_test_fraction=exp.get("donor_test_fraction", 0.2),
        class_types=exp.get("class_types", {}),
        multiclass_groups=exp.get("multiclass_groups", {}),
        multiclass_train_size=exp.get("multiclass_train_size", 40),
        data_path=data_path,
        clips_csv=resolve(exp.get("clips_csv", data_path / CLIPS_CSV.name)),
        results_path=results_path,
        embeddings_path=resolve(exp.get("embeddings_path", results_path / "embeddings")),
        save_classifier=exp.get("save_classifier", False),
        species_sizes=species_sizes,
        run_zeroshot=exp.get("run_zeroshot", True),
    )


def build_partition(manifest, cfg: Config) -> Partition:
    """The single test/train partition shared by every experiment (see make_partition)."""
    group_classes = {c for classes in cfg.multiclass_groups.values() for c in classes}
    min_train = {c: max(cfg.min_train_clips, cfg.multiclass_train_size if c in group_classes else 0)
                 for c in cfg.species}
    return make_partition(manifest, cfg.species, cfg.test_size_per_class, cfg.split_seed,
                          cfg.max_test_recording_clips, min_train, cfg.donor_test_fraction,
                          cfg.low_sample_test_fraction)
