from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BirdNET model training and evaluation."
    )
    parser.add_argument(
        "--datapath", type=str, required=True, help="Path to the audio data directory."
    )
    parser.add_argument(
        "--species_list",
        type=str,
        nargs="+",
        required=True,
        help="List of species to include in the training.",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        choices=["data", "data_5s"],
        required=True,
        help="Type of data to process (3 or 5 seconds).",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--folds", type=int, default=2, help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=25,
        help="Number of samples to use for training.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=2, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Path to save the results.",
    )
    args = parser.parse_args()

    results = main(
        datapath=args.datapath,
        species_list=["human_vocal"],
        datatype="data",
        model_name="BirdNET",
        batch_size=4,
        folds=2,
        training_size=75,
        random_seed=2,
        results_path="/workspaces/non-avian-ml-toy/results",
    )

    if results is None:
        print(
            "Run was skipped due to insufficient samples. No results file was created."
        )
    else:
        print("Run completed successfully.")
