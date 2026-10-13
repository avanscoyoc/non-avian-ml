"""Model files: every model is downloaded on first use into models/ (gitignored) and reused after that.

  BirdNET 2.4 (TFLite FP32), BirdNET 3.0 (TorchScript), Perch 2.0 (CPU SavedModel): birdnet package
  Perch v8 (SavedModel): Kaggle, google/bird-vocalization-classifier, TensorFlow2, version 8
  CNN14 (AudioSet): PANNs checkpoint on Zenodo + AudioSet label list
  ResNet-18 (ImageNet): torchvision weights (TORCH_HOME is set to models/torch in src/paths.py)

scripts/download_models.py calls all of these up front (e.g. before running on a server).
"""
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from src.paths import MODELS  # sets BIRDNET_APP_DATA / TORCH_HOME before the libraries are imported

PERCH_V8_URL = ("https://www.kaggle.com/api/v1/models/google/bird-vocalization-classifier/"
                "tensorFlow2/bird-vocalization-classifier/8/download")
CNN14_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
AUDIOSET_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {dest.name} ...")
        tmp = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def birdnet_v2_4() -> tuple[Path, list[str]]:
    from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4
    path, labels = AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
    return path, list(labels)


def birdnet_v3_0() -> tuple[Path, list[str]]:
    from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0
    path, labels = AcousticPTDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
    return path, list(labels)


def perch_v2() -> tuple[Path, list[str]]:
    from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
    path, labels = AcousticPBDownloaderPerchV2.get_model_path_and_labels("CPU")
    return path, list(labels)


def perch_v8() -> Path:
    model_dir = MODELS / "perch_v8"
    if not (model_dir / "saved_model.pb").exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive = _fetch(PERCH_V8_URL, Path(tmp) / "perch_v8.tar.gz")
            model_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(model_dir, filter="data")
    return model_dir


def cnn14() -> tuple[Path, list[str]]:
    weights = _fetch(CNN14_URL, MODELS / "cnn14" / "Cnn14_mAP=0.431.pth")
    labels = _fetch(AUDIOSET_LABELS_URL, MODELS / "cnn14" / "class_labels_indices.csv")
    return weights, pd.read_csv(labels)["display_name"].tolist()
