import logging
from .evaluate import evaluate_model
from .save_results import ResultsManager
from .config import parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(
    model_name: str,
    species_list: list,
    training_size: int,
    batch_size: int = 32,
    n_folds: int = 5,
    random_seed: int = 42,
    datatype: str = "data",
    datapath: str = "/workspaces/non-avian-ml-toy/data/audio/",
    results_path: str = "/workspaces/non-avian-ml-toy/results",
    gcs_bucket: str = "dse-staff/soundhub",
) -> dict:
    """Run a single experiment with given parameters"""
    try:
        results, fold_scores = evaluate_model(
            model_name=model_name,
            species_list=species_list,
            training_size=training_size,
            batch_size=batch_size,
            n_folds=n_folds,
            random_seed=random_seed,
            datatype=datatype,
            datapath=datapath,
            results_path=results_path,
            gcs_bucket=gcs_bucket,
        )

        logger.info(
            f"Model: {model_name}, Species: {species_list}, Training size: {training_size}"
        )
        for species, roc_auc in results.items():
            logger.info(f"{species}: Mean ROC-AUC = {roc_auc:.4f}")
            for fold_idx, fold_score in enumerate(fold_scores[species]):
                logger.info(f"  Fold {fold_idx + 1}: {fold_score:.4f}")

        return results, fold_scores

    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    # Try to use GCS if credentials are available
    try:
        from google.cloud import storage

        client = storage.Client(project="dse-staff")
        logger.info(f"Connected to GCP project: {client.project}")

        # Test bucket access
        bucket_name = (
            args.gcs_bucket.split("/")[0] if args.gcs_bucket else "dse-staff"
        )
        bucket = client.bucket(bucket_name)
        try:
            bucket.exists()
            logger.info(f"Successfully accessed bucket: {bucket_name}")
            results_manager = ResultsManager(use_gcs=True)
        except Exception as bucket_error:
            logger.warning(f"Bucket access failed: {str(bucket_error)}")
            raise

    except Exception as e:
        logger.warning(f"GCS setup failed: {str(e)}")
        logger.info("Falling back to local storage only")
        results_manager = ResultsManager(use_gcs=False)

    results, fold_scores = run_experiment(
        model_name=args.model,
        species_list=args.species,
        training_size=args.train_size,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        random_seed=args.seed,
        datatype=args.datatype,
        datapath=args.datapath,
        results_path=args.results_path,
        gcs_bucket=args.gcs_bucket,
    )

    # Save results
    results_dict = {
        "mean_roc_auc": results[args.species[0]],
        "fold_scores": {
            i: score for i, score in enumerate(fold_scores[args.species[0]])
        },
    }

    saved_path = results_manager.save_results(
        results_dict=results_dict,
        model_name=args.model,
        species=args.species[0],
        datatype=args.datatype,
        training_size=args.train_size,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        gcs_bucket=args.gcs_bucket,
    )

    logger.info(f"Results saved to: {saved_path}")
