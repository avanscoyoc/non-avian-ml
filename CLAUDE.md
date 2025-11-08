# Claude Instructions: ML Audio Classification Experiment Application

## Project Goal
Build an application that runs multi-factor ML experiments on audio classification models, generating ROC-AUC comparison plots across different training data sizes and species. All models are tuned from existing pretrained models. No models are trained from scratch. 

## Core Requirements

### Data Pipeline
- **Input**: Pull audio data from data folder `data/{species_name}/data/` (pos/neg subfolders)
- **Special case**: Perch model uses `data/{species_name}/data_5s` 
- **Output**: Write results to GCS bucket `results/`
- **Species**: coyote, bullfrog (extensible to more based on config names)
- **Balanced sampling**: Use equal numbers of pos/neg files for training each classifier, do a random draw of samples using 'random_seed' that is configurable
- **Training size limits**: Maximum training size per species is constrained by `min(pos_samples, neg_samples)` to ensure balanced datasets

### ML Experiment Design
- **Models**: 5 models (birdnet, perch, vgg, mobilenet, resnet)
- **Training sizes**: 0-300 samples (configurable intervals, auto-adjusted per species based on available balanced data)
- **Cross-validation**: K-fold with set seeds for reproducibility 'kfold_seed' that is configurable
- **Metrics**: ROC-AUC values with confidence intervals
- **Output visualization**: Per-species charts (x=sample size, y=ROC-AUC, lines=models, error bars=CIs)

### Technical Requirements
- **Platform**: Docker containerized
- **Code quality**: Implement linting (black, flake8, mypy), testing (pytest), proper logging
- **Configuration**: Environment-based config (12-factor app principles)

## Architecture Suggestions
- **Modular design**: Separate data loading, model training, evaluation, results manager, and visualization

## Implementation Guidelines
- Use modern Python practices (3.10+, type hints, dataclasses/pydantic)
- Implement proper error handling and logging (structlog recommended)
- Use efficient ML libraries (scikit-learn, torch, tensorflow as needed)
- Create comprehensive tests and documentation
- Follow security best practices for GCP authentication
- Optimize for both development and production environments

Achieve the most efficient and maintainable solution, DO NOT include print statements, errors, and keep the code as simple and concises as possible and as organized as possible so that I can read it. 

## Deliverables
1. Complete application source code with proper structure in src
2. Documentation (README)

Create a production-ready solution that follows best practices for iterating a module with different arguments in a container and saves results. then i should be able to plot the results with the visualization module according to the Output visualisation suggestion above. 