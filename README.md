# Non-Avian ML Project

A machine learning project for processing non-avian animal sounds using Cloud Run jobs.

## Prerequisites

- [Pixi](https://prefix.dev/) package manager
- Google Cloud SDK
- Access to Google Cloud project `dse-staff`

## Setup & Authentication

1. Clone the repository and change into the directory:
```bash
git clone https://github.com/yourusername/non-avian-ml.git
cd non-avian-ml
```

2. Authenticate with Google Cloud:
```bash
# Login to Google Cloud and set up default credentials
gcloud auth application-default login

# Verify the container image exists in Artifact Registry
gcloud artifacts docker images describe \
    us-central1-docker.pkg.dev/dse-staff/non-avian-ml/model:latest
```

3. Set up service account permissions:
```bash
# Grant Storage Object Viewer role
gcloud projects add-iam-policy-binding dse-staff \
    --member="serviceAccount:cloud-run-jobs@dse-staff.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Grant Storage Bucket Reader role
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

2. Launch the experiment jobs:
```bash
# First, create all jobs
pixi run python src/launch_jobs.py --project-id dse-staff

# Then execute jobs in parallel using gcloud
gcloud beta run jobs list \
  --project dse-staff \
  --region us-central1 \
  --format="value(name)" | \
  xargs -I {} -P 10 gcloud run jobs execute {} \
  --project dse-staff --region us-central1
```

## Monitoring Jobs

Monitor the status of your jobs:
```bash
# List all job executions
gcloud run jobs executions list --project dse-staff --region us-central1

# Get detailed status of a specific job
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

## Troubleshooting

If jobs fail to execute, check:
1. Service account permissions in Google Cloud Console
2. Cloud Run job logs in Google Cloud Console
3. GCS bucket access permissions
4. Container image availability in Artifact Registry

## Contributing

1. Create a new branch
2. Make your changes
3. Submit a pull request

## License

[Add your license information here]