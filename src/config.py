import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Config:
    models: List[str]
    species: List[str]
    training_sizes: Optional[List[int]] = None
    n_folds: int = 5
    n_epochs: int = 20
    test_size_per_class: int = 50
    batch_size: int = 32
    random_seeds: List[int] = None
    kfold_seed: int = 42
    data_path: str = "/workspaces/non-avian-ml/data"
    results_path: str = "/workspaces/non-avian-ml/results"
    datatype: str = "data"
    save_classifier: bool = False
    species_sizes: Optional[Dict[str, Dict[str, int]]] = None
    run_zeroshot: bool = True


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    exp = data["experiments"][0]
    species_sizes = exp.get("species_sizes", None)
    species = exp.get("species", list(species_sizes.keys()) if species_sizes else [])
    return Config(
        models=exp["models"],
        species=species,
        training_sizes=exp.get("training_sizes", None),
        n_folds=exp.get("n_folds", 5),
        n_epochs=exp.get("n_epochs", 20),
        test_size_per_class=exp.get("test_size_per_class", 50),
        batch_size=exp.get("batch_size", 32),
        random_seeds=exp.get("random_seeds", [42]),
        kfold_seed=exp.get("kfold_seed", 42),
        data_path=exp.get("data_path", "/workspaces/non-avian-ml/data"),
        results_path=exp.get("results_path", "/workspaces/non-avian-ml/results"),
        datatype=exp.get("datatype", "data"),
        save_classifier=exp.get("save_classifier", False),
        species_sizes=species_sizes,
        run_zeroshot=exp.get("run_zeroshot", True),
    )
