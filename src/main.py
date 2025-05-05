import os
import argparse

from training_loop import evaluate_model
from config import parse_args


def main(args):
    # Run evaluation
    results, fold_scores = evaluate_model(
        model_name=args.model_name,
        species_list=args.species_list,
        training_size=args.training_size,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        datatype=args.datatype,
        datapath=args.datapath,
        results_path=args.results_path,
        gcs_bucket=args.gcs_bucket,
    )

    print(f"\nFinal Results:")
    print(f"Model: {args.model_name}")
    print(f"Data type: {args.datatype}")
    print(f"Training size: {args.training_size}")
    print("\nResults by species:")
    for species, roc_auc in results.items():
        print(f"{species}: Mean ROC-AUC = {roc_auc:.4f}")
        for fold_idx, fold_score in enumerate(fold_scores[species]):
            print(f"  Fold {fold_idx + 1}: {fold_score:.4f}")

    return results


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
