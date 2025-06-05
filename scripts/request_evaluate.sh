#!/bin/bash

curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "birdnet", 
      "species_list": ["coyote"], 
      "training_size": 10, 
      "batch_size": 32, 
      "n_folds": 2, 
      "random_seed": 42, 
      "datatype": "data", 
      "datapath": "/workspaces/non-avian-ml-toy/data/audio/",
      "results_path": "/workspaces/non-avian-ml-toy/results", 
      "gcs_bucket": "dse-staff/soundhub"
    }'