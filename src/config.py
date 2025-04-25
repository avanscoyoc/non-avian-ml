import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument(
        "--datapath", 
        type=str, 
        default="/workspaces/non-avian-ml-toy/data/audio/",
        help="Path to the audio data directory."
    )
    parser.add_argument(
        "--species_list",
        type=str,
        nargs="+",
        required=True,
        help="List of species to evaluate the model on."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=['resnet', 'mobilenet', 'vgg', 'birdnet', 'perch'],
        required=True,
        help="Name of the model to evaluate."
    )
    parser.add_argument(
        "--datatype",
        type=str,
        choices=["data", "data_5s"],
        default="data",
        help="Type of data to process (3 or 5 seconds)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=10,
        help="Number of samples per class (positive/negative) to use for training."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=2,
        help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="/workspaces/non-avian-ml-toy/results",
        help="Path to save the evaluation results."
    )

    args = parser.parse_args()
    return args