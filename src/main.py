import os
import argparse

from src.training_loop import evaluate_model
from src.config import parse_args


def main(model_name: str, species_list: list, training_size: float, batch_size: int,
         n_folds: int, random_seed: int, datatype: str, datapath: str, results_path: str,
         gcs_bucket: str) -> dict:
    # Run evaluation
    results, fold_scores = evaluate_model(
        model_name=model_name,
        species_list=species_list,
        training_size=training_size,
        batch_size=batch_size,
        n_folds=n_folds,
        random_seed=random_seed,
        datatype=datatype,
        datapath=datapath,
        results_path=results_path,
        gcs_bucket=gcs_bucket,
    )

    print(f"\nFinal Results:")
    print(f"Model: {model_name}")
    print(f"Data type: {datatype}")
    print(f"Training size: {training_size}")
    print("\nResults by species:")
    for species, roc_auc in results.items():
        print(f"{species}: Mean ROC-AUC = {roc_auc:.4f}")
        for fold_idx, fold_score in enumerate(fold_scores[species]):
            print(f"  Fold {fold_idx + 1}: {fold_score:.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
