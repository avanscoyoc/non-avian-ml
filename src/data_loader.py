import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold


def create_train_test_split(data_path, species, datatype, test_size_per_class, seed=42):
    """Create fixed train/test split with deterministic file selection."""
    species_path = Path(data_path) / species / datatype
    all_pos = sorted([str(f) for f in (species_path / "pos").glob("*.wav")])
    all_neg = sorted([str(f) for f in (species_path / "neg").glob("*.wav")])

    if len(all_pos) < test_size_per_class:
        raise ValueError(
            f"{species}: insufficient pos files (need {test_size_per_class}, have {len(all_pos)})"
        )
    if len(all_neg) < test_size_per_class:
        raise ValueError(
            f"{species}: insufficient neg files (need {test_size_per_class}, have {len(all_neg)})"
        )

    rng = random.Random(seed)
    rng.shuffle(all_pos)
    rng.shuffle(all_neg)

    test_pos = all_pos[:test_size_per_class]
    train_pos = all_pos[test_size_per_class:]
    test_neg = all_neg[:test_size_per_class]
    train_neg = all_neg[test_size_per_class:]

    return train_pos, train_neg, test_pos, test_neg


def load_audio_files(training_size, train_pool_pos, train_pool_neg, random_seed=42):
    """Sample balanced training data from train pool."""
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

    combined = list(zip(files, labels))
    rng.shuffle(combined)
    files, labels = zip(*combined)

    return list(files), list(labels)


def preprocess_audio(file_path: str):
    """Convert audio to mel-spectrogram for CNN models."""
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
    db_transform = T.AmplitudeToDB()
    mel = db_transform(mel_transform(waveform))

    if mel.shape[-1] < 128:
        pad_len = 128 - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_len))
    else:
        mel = mel[:, :, :128]

    return mel


def create_kfold_splits(files, labels, n_folds=5, seed=42):
    """Generate stratified K-fold splits."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(files, labels):
        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        yield train_files, train_labels, val_files, val_labels
