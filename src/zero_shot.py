from pathlib import Path
import torch
import numpy as np
from model_loader import load_model
from trainer import cleanup_model
from results import save_run_result


def evaluate_zero_shot(species, model_name, test_files, test_labels, config):
    """
    Evaluate pre-trained model (zero-shot) on test set without any training.
    
    For BirdNET/Perch: Uses their built-in species classifiers
    For CNN models: Uses embeddings with untrained linear layer (random baseline)
    
    Args:
        species: Target species name
        model_name: Model to evaluate
        test_files: List of test audio file paths
        test_labels: List of test labels (1 for positive, 0 for negative)
        config: Config object with results_path, etc.
    
    Returns:
        dict: Result dictionary with test_auc
    """
    # Check if already computed
    run_file = Path(config.results_path) / f"run_{species}_{model_name}_0_0.csv"
    if run_file.exists():
        print(f"  [Zero-shot] Already evaluated (skipping)")
        return None
    
    print(f"  [Zero-shot] Evaluating {model_name.upper()} model...")
    
    try:
        model, device = load_model(model_name)
        
        # Extract embeddings from all test files
        print(f"    Extracting embeddings: {len(test_files)} files")
        embeddings_list = []
        for i, file_path in enumerate(test_files):
            if i > 0 and i % 50 == 0:
                print(f"      Progress: {i}/{len(test_files)}")
            emb = model.extract_embeddings(file_path)
            embeddings_list.append(emb)
        
        X_test = torch.cat(embeddings_list, dim=0).to(device)
        
        # Use a random untrained linear classifier (baseline)
        # This shows performance with zero training samples
        random_classifier = torch.nn.Linear(model.feature_dim, 1).to(device)
        # Don't call any training - just use random initialization
        
        # Get predictions
        with torch.no_grad():
            logits = random_classifier(X_test)
            scores = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(test_labels, scores)
        print(f"    Zero-shot AUC: {test_auc:.4f}")
        
        # Save result with training_size=0
        result = {
            "species": species,
            "model": model_name,
            "training_size": 0,  # Zero training samples
            "n_folds": 0,
            "n_epochs": 0,
            "batch_size": config.batch_size,
            "test_size_per_class": config.test_size_per_class,
            "random_seed": 0,
            "cv_auc_mean": None,
            "test_auc": test_auc,
        }
        save_run_result(result, config.results_path)
        
        # Cleanup
        del random_classifier, X_test, embeddings_list
        cleanup_model(model)
        
        return result
        
    except Exception as e:
        print(f"    ERROR in zero-shot evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_model(model)
        return None