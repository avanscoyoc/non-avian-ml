import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    models: List[str]
    species: List[str]
    training_sizes: List[int]
    n_folds: int = 5
    random_seeds: List[int] = None
    kfold_seed: int = 42
    data_path: str = "/workspaces/non-avian-ml/data"
    results_path: str = "/workspaces/non-avian-ml/results"


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    exp = data["experiments"][0]  # Just use first experiment
    return Config(
        models=exp["models"],
        species=exp["species"],
        training_sizes=exp["training_sizes"],
        n_folds=exp.get("n_folds", 5),
        random_seeds=exp.get("random_seeds", [42]),
        kfold_seed=exp.get("kfold_seed", 42),
        data_path=exp.get("data_path", "/workspaces/non-avian-ml/data"),
        results_path=exp.get("results_path", "/workspaces/non-avian-ml/results"),
    )
