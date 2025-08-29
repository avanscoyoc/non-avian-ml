import os
import logging
from config import parse_args
from evaluate import evaluate_model
from save_results import ResultsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cloud_experiment(
    model_name: str,
    species_list: list,
    training_size: int,
    batch_size: int = 32,
    n_folds: int = 5,
    random_seed: int = 42,
    datatype: str = "data",
) -> dict:
    """Cloud-optimized experiment runner"""
    try:
        # Use Cloud environment variables
        datapath = os.getenv("DATA_PATH", "/tmp/data")
        gcs_bucket = os.getenv("GCS_BUCKET", "dse-staff")
        gcs_prefix = os.getenv("GCS_PREFIX", "soundhub")

        # Run experiment
        results, fold_scores = evaluate_model(
            model_name=model_name,
            species_list=species_list,
            training_size=training_size,
            batch_size=batch_size,
            n_folds=n_folds,
            random_seed=random_seed,
            datatype=datatype,
            datapath=datapath,
            gcs_bucket=gcs_bucket,
        )

        # Save results to GCS
        results_manager = ResultsManager(
            gcs_bucket=gcs_bucket, gcs_prefix=f"{gcs_prefix}/results"
        )
        results_manager.save_results(results, fold_scores)

        return results

    except Exception as e:
        logger.error(f"Cloud experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    run_cloud_experiment(
        model_name=args.model,
        species_list=[args.species],
        training_size=args.train_size,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        random_seed=args.seed,
        datatype=args.datatype,
    )
