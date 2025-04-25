import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath", type=str, default="/workspaces/non-avian-ml-toy/data/audio/", help="Path to the audio data directory."
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
        "--model_name", type=str, default="BirdNET", help="Name of the model to use."
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
        default=10,
        help="Number of samples to use for training.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=1, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/workspaces/non-avian-ml-toy/results",
        help="Path to save the results.",
    )

    args = parser.parse_args()

    return args