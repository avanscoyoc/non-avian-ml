from utils import *

if __name__ == "__main__":
    args = parser.parse_args()
    results = run_model(
        datapath=args.datapath,
        species_list=args.species_list,
        datatype=args.datatype,
        model_name=args.model_name,
        batch_size=args.batch_size,
        folds=args.folds,
        training_size=args.training_size,
        random_seed=args.random_seed,
        results_path=args.results_path,
    )

    if results is None:
        print(
            "Run was skipped due to insufficient samples. No results file was created."
        )
    else:
        print("Run completed successfully.")
