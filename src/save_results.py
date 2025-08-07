import os
import pandas as pd
from datetime import datetime
from google.cloud import storage


class ResultsManager:
    """Handles saving and managing model evaluation results for individual species."""

    def __init__(
        self, results_path="/workspaces/non-avian-ml-toy/results", use_gcs=False
    ):
        """Initialize ResultsManager with results directory path."""
        self.results_path = results_path
        self.use_gcs = use_gcs
        os.makedirs(results_path, exist_ok=True)

    def generate_filename(self, model_name, species, datatype, training_size):
        """Generate standardized filename for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{species}_{model_name}_{datatype}_train{training_size}_{timestamp}.csv"

    def upload_to_gcs(self, local_filepath, bucket_name, blob_name):
        """Upload a file to Google Cloud Storage."""
        try:
            # Try authenticated client first
            try:
                client = storage.Client()
            except Exception:
                # Fall back to anonymous client for public buckets
                client = storage.Client.create_anonymous_client()

            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_filepath)
            print(f"File uploaded to gs://{bucket_name}/{blob_name}")
            return True
        except Exception as e:
            print(f"Warning: Could not upload to GCS - {str(e)}")
            print("Results saved locally only.")
            return False

    def save_results(
        self,
        results_dict,
        model_name,
        species,
        datatype,
        training_size,
        batch_size,
        n_folds,
        gcs_bucket=None,
    ):
        """
        Save evaluation results locally and optionally to GCS.

        Args:
            results_dict: Dictionary containing evaluation results
            model_name: Name of the model used (resnet, mobilenet, vgg, birdnet, perch)
            species: Name of the species evaluated
            datatype: Type of data used (data or data_5s)
            training_size: Number of samples per class used for training
            batch_size: Batch size used during training
            n_folds: Number of folds used in cross-validation
            gcs_bucket: Optional GCS bucket name for uploading results
        """
        filename = self.generate_filename(model_name, species, datatype, training_size)
        filepath = os.path.join(self.results_path, filename)

        # Prepare results data
        results_data = {
            "model_name": model_name,
            "species": species,
            "datatype": datatype,
            "training_size": training_size,
            "batch_size": batch_size,
            "n_folds": n_folds,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mean_roc_auc": results_dict.get("mean_roc_auc", None),
        }

        # Add individual fold scores if available
        fold_scores = results_dict.get("fold_scores", {})
        for fold_idx, score in fold_scores.items():
            results_data[f"fold_{fold_idx + 1}_roc_auc"] = score

        # Create and save DataFrame
        df = pd.DataFrame([results_data])
        df.to_csv(filepath, index=False)
        print(f"Results saved locally to: {filepath}")

        # Upload to GCS if enabled and bucket is specified
        if self.use_gcs and gcs_bucket:
            try:
                bucket_name = gcs_bucket.split("/")[0]
                prefix = "/".join(gcs_bucket.split("/")[1:])
                blob_name = (
                    f"{prefix}/results/{filename}" if prefix else f"results/{filename}"
                )

                if self.upload_to_gcs(filepath, bucket_name, blob_name):
                    return f"gs://{gcs_bucket}/results/{filename}"
            except Exception as e:
                print(f"GCS upload failed: {str(e)}")
                print("Continuing with local save only")

        return filepath

    def load_results(self, species=None, model_name=None):
        """
        Load results from CSV files with optional filtering.

        Args:
            species: Optional species name to filter results
            model_name: Optional model name to filter results

        Returns:
            DataFrame containing all matching results
        """
        all_results = []
        for filename in os.listdir(self.results_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.results_path, filename)
                df = pd.read_csv(filepath)

                if species and df["species"].iloc[0] != species:
                    continue
                if model_name and df["model_name"].iloc[0] != model_name:
                    continue

                all_results.append(df)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def save_batch_results(self, experiments_results, gcs_bucket=None):
        """Save results from multiple experiments"""
        saved_paths = []

        for exp_result in experiments_results:
            if exp_result is None:
                continue

            try:
                path = self.save_results(
                    results_dict=exp_result["results"],
                    model_name=exp_result["config"]["model_name"],
                    species=exp_result["config"]["species_list"][0],
                    datatype=exp_result["config"]["datatype"],
                    training_size=exp_result["config"]["training_size"],
                    batch_size=32,
                    n_folds=5,
                    gcs_bucket=gcs_bucket,
                )
                saved_paths.append(path)
            except Exception as e:
                print(f"Error saving experiment result: {str(e)}")
                continue

        return saved_paths
