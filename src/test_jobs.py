import itertools
import os
from google.cloud import run_v2
from googleapiclient import discovery
import logging
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def create_cloud_run_job(project_id: str, location: str, job_id: str):
    client = run_v2.JobsClient()
    parent = f"projects/{project_id}/locations/{location}"

    try:
        service_account = get_secret(project_id, "cloud-run-service-account")
    except Exception as e:
        service_account = "cloud-run-jobs@dse-staff.iam.gserviceaccount.com"
        logger.warning(f"Using default service account: {str(e)}")

    job = {
        "template": {
            "template": {
                "containers": [
                    {
                        "image": "us-central1-docker.pkg.dev/dse-staff/non-avian-ml/model:latest",
                        "command": ["/bin/sh", "-c"],
                        "args": [
                            "git clone https://github.com/avanscoyoc/non-avian-ml.git &&"
                            "python3 -u non-avian-ml/src/test.py"
                        ],
                        "resources": {"limits": {"cpu": "2", "memory": "8Gi"}},
                        "env": [
                            {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id},
                            {"name": "GCS_BUCKET", "value": "dse-staff"},
                            {"name": "GCS_PREFIX", "value": "soundhub"},
                            {"name": "DATA_PATH", "value": "/tmp/data"},
                            {"name": "PYTHONUNBUFFERED", "value": "1"},
                            {"name": "PYTHONPATH", "value": "/workspaces/non-avian-ml"},
                        ],
                    }
                ],
                "service_account": service_account,
            }
        }
    }

    # Create job with retry
    try:
        operation = client.create_job(
            request={"parent": parent, "job": job, "job_id": job_id}
        )
        return operation.result()
    except Exception as e:
        logger.error(f"Failed to create job: {str(e)}")
        raise


def main(project_id: str, location: str = "us-central1"):
    job_id = f"ml-experiment-{int(time.time())}-{1}"
    job = create_cloud_run_job(
        project_id=project_id,
        location=location,
        job_id=job_id,
    )
    logger.info(f"Launched job: {job.name}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="Cloud Run location")
    args = parser.parse_args()

    main(args.project_id, args.location)