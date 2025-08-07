from model_loader import ModelLoader
from data_processor import DataProcessor as CustomDataLoader
from trainer import *
from save_results import ResultsManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_name, species_list, training_size,
                   batch_size, n_folds, random_seed=42, datatype="data",
                   datapath="/workspaces/non-avian-ml-toy/data/audio/",
                   results_path="/workspaces/non-avian-ml-toy/results",
                   gcs_bucket="dse-staff/soundhub"):
    """Main function to evaluate a model on multiple species datasets."""
    results = {}
    fold_scores_dict = {}

    try:
        for species in species_list:
            logger.info(f"\nEvaluating {species}...")
        # Initialize data loader for current species
        data_loader = CustomDataLoader(
            datapath=datapath,
            species_list=[species],
            datatype=datatype,
            training_size=training_size,
            random_seed=random_seed
        )

        # Load dataset
        df = data_loader.load_data()

        # Initialize model
        model = ModelLoader(
            model_name=model_name,
            num_classes=1  # Binary classification
        )

        # Create appropriate wrapper based on model type
        if model_name.lower() in ['birdnet', 'perch']:
            wrapper = BioacousticsModelWrapper(model.get_model())
        else:
            wrapper = TorchModelWrapper(model.get_model())

        # Initialize trainer and perform k-fold training
        trainer = Trainer(wrapper, n_folds, data_loader)
        mean_roc_auc, fold_scores = trainer.k_fold_train(df, species, batch_size)

        # Store results for current species
        results[species] = mean_roc_auc
        fold_scores_dict[species] = fold_scores

        # Save individual species results
        if results_path:
            results_manager = ResultsManager(results_path)
            species_results = {
                "mean_roc_auc": mean_roc_auc,
                "fold_scores": {i: score for i, score in enumerate(fold_scores)}
            }

            results_manager.save_results(
                results_dict=species_results,
                model_name=model_name,
                species=species,
                datatype=datatype,
                training_size=training_size,
                batch_size=batch_size,
                n_folds=n_folds
            )

    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        raise

    return results, fold_scores_dict