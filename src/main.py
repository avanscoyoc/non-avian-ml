from utils import *

if __name__ == "__main__":
    results = main(
        datapath="/workspaces/non-avian-ml-toy/data/audio/",
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
