import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold


def create_train_test_split(data_path, species, datatype, test_size_per_class, seed=42):
    """Create fixed-size train/test split.

    Args:
        data_path: Root data directory
        species: Species name
        datatype: Subdirectory ('data' or 'data_5s')
        test_size_per_class: Fixed number of test samples per class
        seed: Random seed

    Returns:
        train_pos, train_neg, test_pos, test_neg: Lists of file paths
    """
    species_path = Path(data_path) / species / datatype
    all_pos = sorted([str(f) for f in (species_path / "pos").glob("*.wav")])
    all_neg = sorted([str(f) for f in (species_path / "neg").glob("*.wav")])

    # Verify sufficient data
    if len(all_pos) < test_size_per_class:
        raise ValueError(
            f"{species}: Need {test_size_per_class} pos files for test, "
            f"only have {len(all_pos)}"
        )
    if len(all_neg) < test_size_per_class:
        raise ValueError(
            f"{species}: Need {test_size_per_class} neg files for test, "
            f"only have {len(all_neg)}"
        )

    # Shuffle and split with fixed test size
    rng = random.Random(seed)
    rng.shuffle(all_pos)
    rng.shuffle(all_neg)

    test_pos = all_pos[:test_size_per_class]
    train_pos = all_pos[test_size_per_class:]

    test_neg = all_neg[:test_size_per_class]
    train_neg = all_neg[test_size_per_class:]

    return train_pos, train_neg, test_pos, test_neg


def load_audio_files(training_size, train_pool_pos, train_pool_neg, random_seed=42):
    """Sample balanced data from train pool.

    Args:
        training_size: Number of samples per class
        train_pool_pos: List of positive file paths to sample from
        train_pool_neg: List of negative file paths to sample from
        random_seed: Random seed for sampling

    Returns:
        files, labels: Lists of sampled file paths and corresponding labels
    """
    if len(train_pool_pos) < training_size:
        raise ValueError(
            f"Train pool has {len(train_pool_pos)} pos files, need {training_size}"
        )
    if len(train_pool_neg) < training_size:
        raise ValueError(
            f"Train pool has {len(train_pool_neg)} neg files, need {training_size}"
        )

    rng = random.Random(random_seed)
    pos_sample = rng.sample(train_pool_pos, training_size)
    neg_sample = rng.sample(train_pool_neg, training_size)

    files = pos_sample + neg_sample
    labels = [1] * training_size + [0] * training_size

    # Shuffle
    combined = list(zip(files, labels))
    rng.shuffle(combined)
    files, labels = zip(*combined)

    return list(files), list(labels)


def preprocess_audio(file_path: str):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
    db_transform = T.AmplitudeToDB()
    mel = db_transform(mel_transform(waveform))

    # Pad or truncate to 128
    if mel.shape[-1] < 128:
        pad_len = 128 - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_len))
    else:
        mel = mel[:, :, :128]

    return mel


def create_kfold_splits(files, labels, n_folds=5, seed=42):
    """Generate stratified K-fold splits.

    Yields:
        train_files, train_labels, val_files, val_labels for each fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(files, labels):
        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        yield train_files, train_labels, val_files, val_labels
