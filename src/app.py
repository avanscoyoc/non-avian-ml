from trainer import *
from utils import *


def run_model(
    datapath,
    species_list,
    datatype,
    model_name,
    batch_size,
    folds,
    training_size=None,
    random_seed=42,
    results_path="results",
):
    """Main function to train and evaluate species with optional random sampling."""
    # Load data
    data_loader = DataLoader(
        datapath, species_list, datatype, training_size, random_seed
    )

    try:
        df = data_loader.load_data()
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Skipping this run due to insufficient samples.")
        return None

    # Initialize model and trainer
    model_wrapper = ModelWrapper(model_name, batch_size)
    trainer = Trainer(model_wrapper, folds)

    # Train and evaluate for each species
    results = {}
    for species in species_list:
        avg_roc_auc = trainer.train_and_evaluate_species(df, species)
        results[species] = avg_roc_auc

    # Save results only if we have valid results
    if results:
        result_manager = ResultManager(results_path)
        result_manager.save_results(
            results,
            model_name,
            species_list,
            training_size,
            random_seed,
            batch_size,
            folds,
        )
    
    if results is None:
        print(
            "Run was skipped due to insufficient samples. No results file was created."
        )
    else:
        print("Run completed successfully.")

    return results
