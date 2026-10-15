"""Frozen pretrained encoders, their model files and audio loading.

Every encoder maps 3-s clips to an L2-normalised embedding and, where the model has its own classifier, to raw
label scores (used only for zero-shot). Device policy: PyTorch models use CUDA when available; TensorFlow/TFLite
models always run on the CPU (TF is hidden from the GPU).

Model files are downloaded on first use into cache/models/ (gitignored) and reused after that:
  BirdNET 2.4 (TFLite FP32), BirdNET 3.0 (TorchScript), Perch 2.0 (CPU SavedModel): birdnet package
  Perch v8 (SavedModel): Kaggle, google/bird-vocalization-classifier, TensorFlow2, version 8
  CNN14 (AudioSet): PANNs checkpoint on Zenodo + AudioSet label list
  ResNet-18 (ImageNet): torchvision weights (TORCH_HOME is set to cache/models/torch in core/paths.py)
scripts/download_models.py fetches all of them up front (e.g. before running on a server).
"""
import os
import tarfile
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import models
from torchvision.models import ResNet18_Weights

from core.data import CLIP_SECONDS
from core.paths import MODELS

PERCH_V8_URL = ("https://www.kaggle.com/api/v1/models/google/bird-vocalization-classifier/"
                "tensorFlow2/bird-vocalization-classifier/8/download")
CNN14_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
AUDIOSET_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


# ── Model files ───────────────────────────────────────────────────────────────

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


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_clip(path, sr: int, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Mono float32 at `sr`, centre-cropped or end-padded to exactly `seconds`."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    n = int(round(sr * seconds))
    if len(y) > n:
        start = (len(y) - n) // 2
        y = y[start:start + n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def centre_pad(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Pad with equal silence on both sides (a 3 s clip padded to 5 s gets 1 s on each side)."""
    extra = int(round(sr * seconds)) - len(y)
    if extra <= 0:
        return y
    return np.pad(y, (extra // 2, extra - extra // 2))


# ── Encoders ──────────────────────────────────────────────────────────────────

def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.set_visible_devices([], "GPU")  # keep TF off the GPU (PyTorch uses it)
    except RuntimeError:
        pass  # TF already initialised in this process
    return tf


class Encoder:
    name: str
    # Which model file the encoder uses. The embedding cache stores this and is rebuilt if it changes,
    # so update the string whenever a model file or version changes.
    source: str
    sample_rate: int
    dim: int
    batch_size = 64
    labels: list[str] | None = None  # model's own classes, if any (for zero-shot)
    device = "cpu"

    clip_seconds = CLIP_SECONDS  # input window; clips are centre-cropped or zero-padded to this length

    def _load(self, path) -> np.ndarray:
        return load_clip(path, self.sample_rate, self.clip_seconds)

    def _forward(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """(batch, samples) float32 -> (embeddings, label scores or None)."""
        raise NotImplementedError

    def encode(self, paths: list) -> tuple[np.ndarray, np.ndarray | None]:
        embs, scores = [], []
        start, step = time.time(), max(1, len(paths) // 10)
        # Audio is loaded in the main thread: ~5 ms/clip, and threaded resampling can trip an
        # OpenMP assertion when TensorFlow and PyTorch are both loaded in the process.
        for i in range(0, len(paths), self.batch_size):
            audio = np.stack([self._load(p) for p in paths[i:i + self.batch_size]])
            e, s = self._forward(audio)
            embs.append(e.astype(np.float32))
            if s is not None:
                scores.append(s.astype(np.float32))
            done = min(i + self.batch_size, len(paths))
            if done // step != (done - len(audio)) // step or done == len(paths):
                print(f"    [{self.name}] {done}/{len(paths)} clips ({time.time() - start:.0f}s)", flush=True)
        emb = np.concatenate(embs)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb, (np.concatenate(scores) if scores else None)

    def cleanup(self):
        pass


# TensorFlow / TFLite encoders (CPU)

class BirdNETModel(Encoder):
    """BirdNET 2.4: 1024-D embedding (GLOBAL_AVG_POOL) before the 6,522-class head."""
    name, sample_rate, dim, batch_size = "birdnet", 48000, 1024, 32
    source = "BirdNET 2.4, TFLite FP32 (birdnet package, Zenodo 15050749)"
    EMBEDDING_TENSOR = "model/GLOBAL_AVG_POOL/Mean"

    def __init__(self):
        import ai_edge_litert.interpreter as tflite
        model_path, self.labels = birdnet_v2_4()
        self.model_path = model_path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # preserve_all_tensors "debugging" warning
            self.interpreter = tflite.Interpreter(
                model_path=str(model_path), num_threads=os.cpu_count(),
                experimental_preserve_all_tensors=True,
            )
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.logits_index = self.interpreter.get_output_details()[0]["index"]
        self.embedding_index = next(
            t["index"] for t in self.interpreter.get_tensor_details() if t["name"] == self.EMBEDDING_TENSOR
        )

    def _forward(self, audio):
        embs, logits = [], []
        for clip in audio:
            self.interpreter.set_tensor(self.input_index, clip[None])
            self.interpreter.invoke()
            embs.append(self.interpreter.get_tensor(self.embedding_index)[0].copy())
            logits.append(self.interpreter.get_tensor(self.logits_index)[0].copy())
        return np.stack(embs), np.stack(logits)

    def cleanup(self):
        del self.interpreter


class _PaddedTFEncoder(Encoder):
    """5-s TF models: centre-pad 3-s clips; fixed batch shape avoids XLA recompiles.
    Batch 8 is as fast as 16-32 on CPU and keeps peak memory near 3 GB."""
    sample_rate, window_s, batch_size = 32000, 5.0, 8

    def _load(self, path):
        return centre_pad(load_clip(path, self.sample_rate, self.clip_seconds), self.sample_rate, self.window_s)

    def _call(self, batch) -> dict:
        raise NotImplementedError

    def _forward(self, audio):
        n = len(audio)
        if n < self.batch_size:
            audio = np.concatenate([audio, np.zeros((self.batch_size - n, audio.shape[1]), np.float32)])
        out = self._call(self.tf.constant(audio))
        scores = out["label"].numpy()[:n] if self.labels is not None else None
        return out["embedding"].numpy()[:n], scores


class PerchModel(_PaddedTFEncoder):
    """Perch v8 (bird-vocalization-classifier, Perch 1.0 family): 1280-D embedding; bird labels only."""
    name, dim = "perch", 1280
    source = "Perch v8, Kaggle google/bird-vocalization-classifier/TensorFlow2/8"

    def __init__(self):
        self.tf = _import_tf()
        import tensorflow_hub as hub
        self.model_path = perch_v8()
        self._model = hub.load(str(self.model_path))
        self._call = lambda x: self._model.infer_tf(x)

    def cleanup(self):
        del self._model, self._call


class Perch2Model(_PaddedTFEncoder):
    """Perch 2.0 (CPU SavedModel via the birdnet package): 1536-D embedding, 14,795 labels (iNat + FSD50K)."""
    name, dim = "perch2", 1536
    source = "Perch 2.0, CPU SavedModel (birdnet package)"

    def __init__(self):
        self.tf = _import_tf()
        self.model_path, self.labels = perch_v2()
        self._model = self.tf.saved_model.load(str(self.model_path))
        fn = self._model.signatures["serving_default"]
        self._call = lambda x: fn(inputs=x)

    def cleanup(self):
        del self._model, self._call


# PyTorch encoders (CUDA if available)

class _TorchEncoder(Encoder):
    batch_size = 32

    def __init__(self, device=None):
        self.device = device or torch_device()

    def cleanup(self):
        for attr in ("model", "backbone"):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BirdNET3Model(_TorchEncoder):
    """BirdNET 3.0 (preview3.1, TorchScript via the birdnet package): 1280-D embedding, >11k labels."""
    name, sample_rate, dim = "birdnet3", 32000, 1280
    source = "BirdNET 3.0 preview3.1, TorchScript FP32 (birdnet package)"

    def __init__(self, device=None):
        super().__init__(device)
        self.model_path, self.labels = birdnet_v3_0()
        self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()

    @torch.inference_mode()
    def _forward(self, audio):
        scores, emb = self.model(torch.from_numpy(audio).to(self.device))
        return emb.cpu().numpy(), scores.cpu().numpy()


class CNN14Model(_TorchEncoder):
    """PANNs CNN14 (AudioSet): 2048-D embedding, 527 AudioSet labels."""
    name, sample_rate, dim = "cnn14", 32000, 2048
    source = "PANNs CNN14 Cnn14_mAP=0.431.pth (Zenodo 3987831)"

    def __init__(self, device=None):
        super().__init__(device)
        from core.cnn14 import Cnn14
        self.model_path, self.labels = cnn14()
        self.model = Cnn14.from_checkpoint(self.model_path).to(self.device)

    @torch.inference_mode()
    def _forward(self, audio):
        scores, emb = self.model(torch.from_numpy(audio).to(self.device))
        return emb.cpu().numpy(), scores.cpu().numpy()


class ResNetModel(_TorchEncoder):
    """ResNet-18 (ImageNet) on log-mel spectrograms, fully pretrained and in eval mode: 512-D embedding.

    The 16 kHz, 64-mel dB spectrogram of the whole 3-s clip is scaled per clip to [0, 1],
    repeated to 3 channels and ImageNet-normalised, so the pretrained first layer is kept.
    """
    name, sample_rate, dim = "resnet", 16000, 512
    source = "torchvision ResNet-18 IMAGENET1K_V1"
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device=None):
        super().__init__(device)
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval().to(self.device)
        self.melspec = nn.Sequential(T.MelSpectrogram(sample_rate=16000, n_mels=64), T.AmplitudeToDB()).to(self.device)
        self.mean, self.std = self.MEAN.to(self.device), self.STD.to(self.device)

    @torch.inference_mode()
    def _forward(self, audio):
        mel = self.melspec(torch.from_numpy(audio).to(self.device)).unsqueeze(1)  # (B, 1, 64, 241)
        lo = mel.amin(dim=(2, 3), keepdim=True)
        hi = mel.amax(dim=(2, 3), keepdim=True)
        x = ((mel - lo) / (hi - lo + 1e-8)).expand(-1, 3, -1, -1)
        return self.backbone((x - self.mean) / self.std).cpu().numpy(), None


ENCODERS = {e.name: e for e in (BirdNETModel, BirdNET3Model, PerchModel, Perch2Model, CNN14Model, ResNetModel)}
MODEL_NAMES = list(ENCODERS)


def load_model(model_name: str, clip_seconds: float | None = None) -> Encoder:
    """Load a frozen encoder. clip_seconds: input window if not the default 3 s (BirdNET 3.0 accepts any length)."""
    model = ENCODERS[model_name]()
    if clip_seconds:
        model.clip_seconds = clip_seconds
    print(f"[{model_name}] loaded on {model.device} (embedding dim {model.dim}, {model.clip_seconds:g} s input)")
    return model
