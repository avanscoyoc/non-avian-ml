# Non-Avian ML Project

A machine learning project for comparing model type performance on non-avian animal sounds using Cloud Run jobs.

## Prerequisites

- [Pixi](https://prefix.dev/) package manager
- Google Cloud SDK
- Access to Google Cloud project `dse-staff`

## Setup & Authentication

1. Clone the repository, set directory:
```bash
git clone https://github.com/yourusername/non-avian-ml.git
cd non-avian-ml
```

2. Authenticate with Google Cloud:

Login to Google Cloud and set up default credentials
```bash
gcloud auth application-default login
```
Verify the container image exists in Artifact Registry
```bash
gcloud artifacts docker images describe \
    us-central1-docker.pkg.dev/dse-staff/non-avian-ml/model:latest
```

3. Set up service account permissions (storage viewer and bucket reader):
```bash
gcloud projects add-iam-policy-binding dse-staff \
    --member="serviceAccount:cloud-run-jobs@dse-staff.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer" && \
gcloud projects add-iam-policy-binding dse-staff \
    --member="serviceAccount:cloud-run-jobs@dse-staff.iam.gserviceaccount.com" \
    --role="roles/storage.bucketViewer"
```

## Running the Project

1. Initialize the Pixi environment:
(Not necessary if opened in Container in VS Code)
```bash
pixi install
```
Option A. Launch single experiment example: 
```bash
export DATA_DIR="/tmp/data/audio" && \
mkdir -p "${DATA_DIR}/coyote/data/"{pos,neg} && \
gsutil -m cp "gs://dse-staff/soundhub/data/audio/coyote/data/pos/*" "${DATA_DIR}/coyote/data/pos/" && \
gsutil -m cp "gs://dse-staff/soundhub/data/audio/coyote/data/neg/*" "${DATA_DIR}/coyote/data/neg/" && \
pixi run python src/main.py --model mobilenet --species coyote --train_size 10 --seed 1 --datatype data --batch_size 32 --n_folds 2 --datapath "${DATA_DIR}" && \
rm -rf "${DATA_DIR}" /tmp/results
```
Option B. Launch multi-model and species experiment to Cloud Run:
```bash
pixi run python src/launch_jobs.py --project-id dse-staff && \
gcloud beta run jobs list \
  --project dse-staff \
  --region us-central1 \
  --format="value(name)" | \
  xargs -I {} -P 10 gcloud run jobs execute {} \
  --project dse-staff --region us-central1
```

## Configuration

Experiment configurations can be modified in `src/launch_jobs.py`:

```python
EXPERIMENT_CONFIG = {
    "models": ["mobilenet", "birdnet"],
    "species": ["coyote", "bullfrog"],
    "train_sizes": [10, 20],
    "seeds": [1, 2],
    "datatypes": ["data"],
    "batch_sizes": [32],
    "n_folds": [5],
}
```

## Monitoring Jobs

Monitor the status of your jobs:
```bash
gcloud run jobs executions list --project dse-staff --region us-central1
```
```bash
gcloud run jobs describe JOB_NAME --project dse-staff --region us-central1
```

## Project Structure

```
.
├── src/
│   ├── launch_jobs.py    # Cloud Run job launcher
│   ├── main.py          # Main experiment runner
│   └── ...
├── pixi.toml           # Pixi package configuration
└── README.md
```

## Troubleshooting

If jobs fail to execute, check:
1. Service account permissions in Google Cloud Console
2. Cloud Run job logs in Google Cloud Console
3. GCS bucket access permissions
4. Container image availability in Artifact Registry

## License

to add...