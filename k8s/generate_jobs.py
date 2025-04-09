import itertools
import yaml
import os


def generate_jobs(max_parallel=10):
    params = {
        "species_list": [
            ["bullfrog"],
            ["coyote"],
            ["engine"],
            ["field_cricket"],
            ["human_vocal"],
            ["pacific_chorus_frog"],
            ["woodhouses_toad"],
        ],
        "training_size": [50, 75, 100],
        "model_name": ["BirdNET", "RESNET", "MOBILENET"],
        "batch_size": [4, 8],
        "folds": [2, 3, 5],
    }

    # Create output directory
    os.makedirs("jobs", exist_ok=True)

    # Generate all combinations
    keys = params.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*params.values())]

    # Create a job batch controller
    batch_job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "ml-training-batch"},
        "spec": {
            "parallelism": max_parallel,  # Control parallel jobs
            "completions": len(combinations),
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "training",
                            "image": "your-registry/non-avian-ml:latest",
                            "command": ["python", "src/main.py"],
                            "args": [],  # Will be set by job index
                            "volumeMounts": [
                                {"name": "data-volume", "mountPath": "/data"},
                                {"name": "results-volume", "mountPath": "/results"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "data-volume",
                            "persistentVolumeClaim": {"claimName": "audio-data-pvc"},
                        },
                        {
                            "name": "results-volume",
                            "persistentVolumeClaim": {"claimName": "results-pvc"},
                        },
                    ],
                    "restartPolicy": "Never",
                }
            },
        },
    }

    # Write batch job YAML
    with open("jobs/batch-job.yaml", "w") as f:
        yaml.dump(batch_job, f)

#kubectl apply -f jobs/batch-job.yaml
#kubectl get jobs ml-training-batch
#kubectl get pods -l job-name=ml-training-batch