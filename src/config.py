import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate audio model")

    # Required parameters
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet", "mobilenet", "vgg", "birdnet", "perch"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        required=True,
        help="Species to analyze",
    )
    parser.add_argument(
        "--train_size", type=int, required=True, help="Training size to use"
    )

    # Optional parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--datatype",
        type=str,
        choices=["data", "data_5s"],
        default="data",
        help="Type of data to process",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--datapath",
        type=str,
        default="/workspaces/non-avian-ml-toy/data/audio/",
        help="Path to audio data",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/workspaces/non-avian-ml-toy/results",
        help="Path to save results",
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default="dse-staff/soundhub",
        help="GCS bucket for results",
    )

    return parser.parse_args()
