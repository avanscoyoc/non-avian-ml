import os
import glob
import random
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from google.cloud import storage
import logging
import tempfile
import shutil


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


class LazyAudioDataset(AudioDataset):
    """Memory efficient dataset that loads audio files on demand."""

    def __init__(self, files, labels, transform=None):
        super().__init__(files, labels)
        self.transform = transform
        self._loaded_items = {}  # Small LRU cache for frequently accessed items

    def __getitem__(self, idx):
        if idx in self._loaded_items:
            return self._loaded_items[idx]

        # Load audio file only when needed
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        # Cache result with limited size
        if len(self._loaded_items) > 100:  # Keep only last 100 items
            self._loaded_items.clear()
        self._loaded_items[idx] = (waveform, self.labels[idx])

        return waveform, self.labels[idx]


class DataProcessor:
    """Handles loading and preprocessing of data with caching."""

    def __init__(
        self,
        datapath,
        species_list,
        datatype="data",
        training_size=None,
        random_seed=42,
        gcs_bucket="dse-staff",
    ):
        self.datapath = datapath
        self.gcs_bucket = gcs_bucket
        self.species_list = species_list
        self.datatype = datatype  # Keeping datatype for data vs data_5s
        self.training_size = training_size
        self._cache = {}  # Cache for processed data
        self._downloaded_species = set()  # Track downloaded species
        self.temp_dirs = []  # Track temp directories for cleanup

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        random.seed(random_seed)
        os.makedirs(datapath, exist_ok=True)

    def _create_temp_dir(self):
        """Create a temporary directory that will be cleaned up."""
        temp_dir = tempfile.mkdtemp(prefix="audio_data_")
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def download_gcs_folder(self, species):
        """Downloads species data from GCS bucket."""
        try:
            # Construct GCS path
            gcs_prefix = f"soundhub/data/audio/{species}/{self.datatype}"

            # Create temp directory for this species
            temp_species_dir = os.path.join(self.datapath, species, self.datatype)
            os.makedirs(temp_species_dir, exist_ok=True)

            # Download files
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_bucket)

            # List and download blobs
            blobs = list(bucket.list_blobs(prefix=gcs_prefix))
            if not blobs:
                raise ValueError(
                    f"No files found in gs://{self.gcs_bucket}/{gcs_prefix}"
                )

            for blob in blobs:
                # Get relative path by removing prefix
                relative_path = blob.name[len(gcs_prefix) :].lstrip("/")
                local_path = os.path.join(temp_species_dir, relative_path)

                # Create directories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)

            return True

        except Exception as e:
            logger.error(f"Failed to download {species}: {str(e)}")
            return False

    def load_species_data(self, species):
        """Load species data with caching for different training sizes."""
        cache_key = f"{species}_{self.datatype}_{self.training_size}"

        # Check cache first
        if cache_key in self._cache:
            self.logger.info(
                f"Using cached data for {species}/{self.datatype} ({self.training_size} samples)"
            )
            return self._cache[cache_key]

        # Download if needed
        if not self.download_gcs_folder(species):
            raise RuntimeError(f"Failed to download data for {species}/{self.datatype}")

        # Get file paths
        pos_path = os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
        neg_path = os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")

        pos_files = sorted(glob.glob(pos_path))  # Sort for deterministic sampling
        neg_files = sorted(glob.glob(neg_path))

        if not pos_files or not neg_files:
            raise ValueError(
                f"No audio files found for {species} at {pos_path} or {neg_path}"
            )

        # Sample if training_size specified
        if self.training_size is not None:
            if (
                len(pos_files) < self.training_size
                or len(neg_files) < self.training_size
            ):
                raise ValueError(
                    f"Insufficient samples for {species}: "
                    f"pos={len(pos_files)}, neg={len(neg_files)}, "
                    f"requested={self.training_size}"
                )

            pos_files = random.sample(pos_files, self.training_size)
            neg_files = random.sample(neg_files, self.training_size)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "files": pos_files + neg_files,
                species: [1] * len(pos_files) + [0] * len(neg_files),
                "noise": [0] * len(pos_files) + [1] * len(neg_files),
            }
        )

        # Cache results
        self._cache[cache_key] = df
        return df

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

    def __del__(self):
        """Cleanup temporary directories on object destruction."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {temp_dir}: {str(e)}")
