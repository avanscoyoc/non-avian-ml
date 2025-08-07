import itertools
import os
from google.cloud import run_v2
from googleapiclient import discovery
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for experiments
EXPERIMENT_CONFIG = {
    "models": ["mobilenet", "birdnet"],
    "species": ["coyote", "bullfrog"],
    "train_sizes": [10, 20],
    "seeds": [1],
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


def get_secret(project_id: str, secret_name: str):
    """Get secret using Google API client instead of Cloud Secret Manager"""
    try:
        service = discovery.build("secretmanager", "v1", cache_discovery=False)
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = service.projects().secrets().versions().access(name=name).execute()
        return base64.b64decode(response["payload"]["data"]).decode("UTF-8")
    except Exception as e:
        logger.warning(f"Failed to get secret: {str(e)}")
        return os.getenv(
            "CLOUD_RUN_SERVICE_ACCOUNT",
            "cloud-run-jobs@dse-staff.iam.gserviceaccount.com",
        )


def create_cloud_run_job(project_id: str, location: str, args: list, job_id: str):
    """Create a Cloud Run job for a single experiment"""
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"

    # Get service account from Secret Manager
    try:
        service_account = get_secret(project_id, "cloud-run-service-account")
    except Exception as e:
        # Fallback to environment variable if secret not available
        service_account = os.getenv(
            "CLOUD_RUN_SERVICE_ACCOUNT",
            "cloud-run-jobs@dse-staff.iam.gserviceaccount.com",  # default fallback
        )
        logger.warning(f"Using fallback service account: {str(e)}")

    # Create job configuration
    job = {
        "template": {
            "template": {
                "containers": [
                    {
                        "image": (
                            "us-central1-docker.pkg.dev/dse-staff/"
                            "non-avian-ml-toy/model:latest"
                        ),
                        "args": args,
                        "resources": {"limits": {"cpu": "2", "memory": "8Gi"}},
                        "env": [
                            {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id},
                            {"name": "GCS_BUCKET", "value": "dse-staff"},
                            {"name": "GCS_PREFIX", "value": "soundhub"},
                            {
                                "name": "DATA_PATH",
                                "value": "/tmp/data/audio",
                            },  # Use temp directory
                        ],
                    }
                ],
                "service_account": service_account,
            }
        }
    }

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
                project_id=project_id,
                location=location,
                args=cmd,
                job_id=job_id,
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
