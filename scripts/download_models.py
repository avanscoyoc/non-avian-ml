"""
download_models.py

Fetches the models that are not committed to the repo into models/ (gitignored):
  - BirdNET 3.0 (preview3.1, TorchScript) and Perch 2.0 (CPU SavedModel), via the birdnet package
  - PANNs CNN14 (AudioSet) weights + AudioSet label list

BirdNET 2.4 (TFLite) and Perch v8 are already in models/.

Usage:
    pixi run python scripts/download_models.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import MODELS  # noqa: E402  (sets BIRDNET_APP_DATA before birdnet is imported)

CNN14_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
AUDIOSET_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists: {dest.relative_to(MODELS.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main():
    from birdnet.acoustic.models.perch_v2.pb import AcousticPBDownloaderPerchV2
    from birdnet.acoustic.models.v3_0.pt import AcousticPTDownloaderV3_0

    print("BirdNET 3.0 (pt):")
    path, labels = AcousticPTDownloaderV3_0.get_model_path_and_labels("en_us", "fp32")
    print(f"  {path.name}: {len(labels)} labels")

    print("Perch 2.0 (cpu):")
    path, labels = AcousticPBDownloaderPerchV2.get_model_path_and_labels("CPU")
    print(f"  {path.name}: {len(labels)} labels")

    print("CNN14 (AudioSet):")
    fetch(CNN14_URL, MODELS / "cnn14" / "Cnn14_mAP=0.431.pth")
    fetch(AUDIOSET_LABELS_URL, MODELS / "cnn14" / "class_labels_indices.csv")
    print("Done.")


if __name__ == "__main__":
    main()
