import os
import glob
import random
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch


class AudioDataset:
    """Base class for audio datasets"""
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)


class TorchAudioDataset(AudioDataset):
    """Dataset for PyTorch models (ResNet, MobileNet, VGG)"""
    def __init__(self, files, labels, sample_rate=16000,
                 n_mels=64, max_len=128):
        super().__init__(files, labels)
        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, 
                                              n_mels=n_mels)
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
            mel = mel[:, :, :self.max_len]

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
        self, datapath, species_list, datatype="data",
        training_size=None, random_seed=42
    ):
        self.datapath = datapath
        self.species_list = species_list
        self.datatype = datatype
        self.training_size = training_size
        random.seed(random_seed)

    def sample_files(self, files, size, species, file_type="positive"):
        """Helper method to sample files with proper error handling."""
        if len(files) < size:
            print(
                f"Warning: Requested {size} {file_type} samples but only found {len(files)} for {species}"
            )
            return files
        return random.sample(files, size)

    def load_species_data(self, species):
        """Load data for a single species with optional random sampling."""
        pos_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "pos", "*.wav")
        )
        neg_files = glob.glob(
            os.path.join(self.datapath, species, self.datatype, "neg", "*.wav")
        )

        # Check if we have enough samples before proceeding
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
