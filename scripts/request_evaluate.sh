#!/bin/bash

curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test_model", "species_list": ["species1", "species2"], "training_size": 0.8, "batch_size": 32, "n_folds": 5, "random_seed": 42, "datatype": "features"}'