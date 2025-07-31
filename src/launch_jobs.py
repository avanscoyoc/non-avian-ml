import itertools
from google.cloud import run_v2
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for experiments
EXPERIMENT_CONFIG = {
    "models": ["mobilenet", "birdnet"],
    "species": ["coyote", "bullfrog", "woodhouses_toad"],
    "train_sizes": [10, 20, 30],
    "seeds": [1, 2, 3],
    "datatypes": ["data"],
    "batch_sizes": [32],
    "n_folds": [5],
}


def generate_job_commands():
    """Generate all experiment combinations as command line arguments"""
    combinations = itertools.product(
        EXPERIMENT_CONFIG["models"],
        EXPERIMENT_CONFIG["species"],
        EXPERIMENT_CONFIG["train_sizes"],
        EXPERIMENT_CONFIG["seeds"],
        EXPERIMENT_CONFIG["datatypes"],
        EXPERIMENT_CONFIG["batch_sizes"],
        EXPERIMENT_CONFIG["n_folds"],
    )

    commands = []
    for model, species, train_size, seed, datatype, batch_size, n_folds in combinations:
        # Generate list of arguments for the container
        cmd = [
            "--model",
            model,
            "--species",
            species,
            "--train_size",
            str(train_size),
            "--seed",
            str(seed),
            "--datatype",
            datatype,
            "--batch_size",
            str(batch_size),
            "--n_folds",
            str(n_folds),
            "--gcs_bucket",
            "dse-staff/soundhub",
        ]
        commands.append(cmd)

    return commands


def create_cloud_run_job(project_id: str, location: str, args: list, job_id: str):
    """Create a Cloud Run job for a single experiment"""
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"

    # Create job template
    job = run_v2.Job()
    job.template = run_v2.ExecutionTemplate()

    # Container configuration
    container = run_v2.Container(
        image="us-central1-docker.pkg.dev/dse-staff/non-avian-ml-toy/model:latest",
        args=args,
        resources=run_v2.ResourceRequirements(limits={"cpu": "2", "memory": "8Gi"}),
    )
    job.template.containers = [container]

    # Service account for GCS access
    service_account = "service-422941045810@gs-project-accounts.iam.gserviceaccount.com"
    job.template.service_account = service_account

    # Create job with retry
    operation = client.create_job(parent=parent, job=job, job_id=job_id)
    return operation.result()


def main(project_id: str, location: str = "us-central1"):
    """Launch all experiment jobs to Cloud Run"""
    commands = generate_job_commands()
    logger.info(f"Generated {len(commands)} experiment commands")

    for i, cmd in enumerate(commands):
        try:
            job_id = f"ml-experiment-{int(time.time())}-{i}"
            logger.info(f"Launching job {i + 1}/{len(commands)}")
            logger.info(f"Command: {' '.join(cmd)}")

            job = create_cloud_run_job(
                project_id=project_id, location=location, args=cmd, job_id=job_id
            )
            logger.info(f"Launched job: {job.name}")

            # Add a small delay between job submissions
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error launching job {i + 1}: {str(e)}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Cloud Run location")
    args = parser.parse_args()

    main(args.project_id, args.location)
