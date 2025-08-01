import os
import glob
import random
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch
from google.cloud import storage


class AudioDataset:
    """Base class for audio datasets"""

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)


class TorchAudioDataset(AudioDataset):
    """Dataset for PyTorch models (ResNet, MobileNet, VGG)"""

    def __init__(self, files, labels, sample_rate=16000, n_mels=64, max_len=128):
        super().__init__(files, labels)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db_transform = T.AmplitudeToDB()
        self.max_len = max_len

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)

        mel = self.db_transform(self.mel_transform(waveform))

        if mel.shape[-1] < self.max_len:
            pad_len = self.max_len - mel.shape[-1]
            mel = torch.nn.functional.pad(mel, (0, pad_len))
        else:
            mel = mel[:, :, : self.max_len]

        return mel, self.labels[idx]


class BioacousticsDataset(AudioDataset):
    """Dataset for bioacoustics models (BirdNET, Perch)"""

    def __init__(self, files, labels, sample_rate=48000):
        super().__init__(files, labels)
        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform, self.labels[idx]


class DataProcessor:
    """Handles loading and preprocessing of data."""

    def __init__(
        self,
        datapath,
        species_list,
        datatype="data",
        training_size=None,
        random_seed=42,
        gcs_bucket="dse-staff",
        gcs_prefix="soundhub",
    ):
        self.datapath = datapath
        self.species_list = species_list
        self.datatype = datatype
        self.training_size = training_size
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        random.seed(random_seed)

        # Ensure data directory exists
        os.makedirs(datapath, exist_ok=True)

    def download_data_from_gcs(self, species):
        """Download species data from GCS"""
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_bucket)

            # Define paths
            gcs_species_path = f"{self.gcs_prefix}/audio/{species}/{self.datatype}"
            local_species_path = os.path.join(self.datapath, species, self.datatype)

            # Create local directories
            os.makedirs(os.path.join(local_species_path, "pos"), exist_ok=True)
            os.makedirs(os.path.join(local_species_path, "neg"), exist_ok=True)

            # Download files
            blobs = bucket.list_blobs(prefix=gcs_species_path)
            for blob in blobs:
                if blob.name.endswith(".wav"):
                    local_path = os.path.join(
                        self.datapath, os.path.relpath(blob.name, self.gcs_prefix)
                    )
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
                    logger.info(f"Downloaded {blob.name} to {local_path}")

            return True
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return False

    def load_species_data(self, species):
        """Load data for a single species with GCS fallback"""
        # First try to find local files
        pos_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
        )
        neg_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")
        )

        # If no local files found, try downloading from GCS
        if not pos_files or not neg_files:
            logger.info(f"No local data found for {species}, downloading from GCS...")
            if not self.download_data_from_gcs(species):
                raise ValueError(f"Could not download data for {species} from GCS")

            # Try loading files again after download
            pos_files = glob.glob(
                os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
            )
            neg_files = glob.glob(
                os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")
            )

        # Verify we have enough samples
        if self.training_size is not None:
            if (
                len(pos_files) < self.training_size
                or len(neg_files) < self.training_size
            ):
                raise ValueError(
                    f"Insufficient samples for {species}: "
                    f"Found {len(pos_files)} positive and {len(neg_files)} negative samples, "
                    f"but {self.training_size} samples were requested."
                )

        if self.training_size is not None:
            min_samples = min(len(pos_files), len(neg_files))
            training_size = min(self.training_size, min_samples)
            pos_files = self.sample_files(pos_files, training_size, species, "positive")
            neg_files = self.sample_files(neg_files, training_size, species, "negative")

            print(
                f"Using {len(pos_files)} positive and {len(neg_files)} negative samples for {species}"
            )

        all_files = pos_files + neg_files
        encoding_pos_files = [1] * len(pos_files) + [0] * len(neg_files)
        encoding_neg_files = [0] * len(pos_files) + [1] * len(neg_files)

        return pd.DataFrame(
            {
                "files": all_files,
                species: encoding_pos_files,
                "noise": encoding_neg_files,
            }
        )

    def load_data(self):
        """Load data for all species."""
        print("Loading dataset...")
        df_each_species = {}
        for species in self.species_list:
            df_each_species[species] = self.load_species_data(species)

        all_species = pd.concat(df_each_species.values(), axis=0)
        all_species.fillna(0, inplace=True)
        all_species.set_index("files", inplace=True)
        all_species = all_species.astype(int)
        print("Dataset loaded:")
        print(all_species.sum("index"))
        print(all_species.head())
        return all_species

    def get_dataset(self, df, species, model_type="torch"):
        """
        Returns appropriate dataset based on model type
        Args:
            df: DataFrame containing file paths and labels
            species: Species name for labels
            model_type: Either 'torch' or 'bioacoustics'
        """
        file_list = df.index.tolist()
        labels = df[species].tolist()

        if model_type.lower() == "torch":
            return TorchAudioDataset(file_list, labels)
        elif model_type.lower() == "bioacoustics":
            return df
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def download_from_gcs(self, bucket_name="dse-staff", prefix="soundhub"):
        """Download dataset from GCS at runtime"""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Create local data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Download all blobs with the given prefix
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if not blob.name.endswith("/"):  # Skip directories
                local_path = blob.name
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
