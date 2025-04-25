import os
import glob
import random
import pandas as pd


class DataLoader:
    """Handles loading and preprocessing of data."""

    def __init__(
        self, datapath, species_list, datatype, training_size=None, random_seed=42
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
