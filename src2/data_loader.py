import random
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold


def load_audio_files(
    data_path: str,
    species: str,
    training_size: int,
    datatype: str = "data",
    random_seed: int = None,
):
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    species_path = Path(data_path) / species / datatype
    pos_files = list((species_path / "pos").glob("*.wav"))
    neg_files = list((species_path / "neg").glob("*.wav"))

    # Sample balanced data
    pos_sample = random.sample(pos_files, training_size)
    neg_sample = random.sample(neg_files, training_size)

    files = [str(f) for f in pos_sample + neg_sample]
    labels = [1] * training_size + [0] * training_size

    # Shuffle
    combined = list(zip(files, labels))
    random.shuffle(combined)
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


def create_kfold_splits(files, labels, n_folds=5, seed=1):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(files, labels))
