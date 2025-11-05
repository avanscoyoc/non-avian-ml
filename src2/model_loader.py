import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import librosa
import ai_edge_litert.interpreter as tflite


class BirdNETModel:
    """BirdNET feature extractor using the actual TFLite model.

    - Input: 3 s audio at 48 kHz (shape [1, 144000])
    - Embeddings: 1024-d vector from GLOBAL_AVG_POOL/Mean tensor
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.feature_dim = 1024
        self.sample_rate = 48000
        self.length_s = 3.0
        self.interpreter = None
        self.input_index = None
        self.embedding_index = None

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # I/O details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]

        # Locate the 1024-d embedding tensor (GLOBAL_AVG_POOL/Mean)
        for td in self.interpreter.get_tensor_details():
            name = td.get("name", "").lower()
            shape = td.get("shape", [])
            if "global_avg_pool/mean" in name or (len(shape) == 2 and shape[1] == 1024):
                self.embedding_index = td["index"]
                break

    def _load_audio(self, path: str) -> np.ndarray:
        # Load mono audio at required sample rate and pad/trim to 3 s
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        target_len = int(self.sample_rate * self.length_s)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        return y.astype(np.float32)

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        # Prepare input
        audio = self._load_audio(audio_file_path)
        inp = np.expand_dims(audio, axis=0)

        # Inference
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()

        if self.embedding_index is not None:
            emb = self.interpreter.get_tensor(self.embedding_index)
            vec = emb.reshape(-1).astype(np.float32)
        else:
            # Fallback to logits (still from the real model)
            logits = self.interpreter.get_tensor(self.output_details[0]["index"])
            vec = logits[0][: self.feature_dim].astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm
        return torch.from_numpy(vec).unsqueeze(0)


class PerchModel:
    """Perch feature extractor via Google Research 'chirp' inference.

    Uses official Perch embeddings (EfficientNet, 1280-d). Requires
    installing from the Perch repo (google-research/perch).
    """

    def __init__(self):
        self.window_size_s = 5.0
        self.sample_rate = 32000
        self.feature_dim = 1280

        try:
            from chirp.inference import embed_lib

            # Load default Perch model from the zoo
            # Model key may vary; see Perch README for options.
            self._embed = embed_lib.get_embedding_model(
                model_key="perch_8",
                frontend="pcen_mel",
            )
            self._embed_model = self._embed.embed_fn
            self._wav_to_examples = self._embed.wav_to_examples
        except Exception as e:
            raise RuntimeError(
                "Perch needs 'chirp' inference. Install from GitHub "
                "(google-research/perch) or use their Docker image. "
                "Error: " + str(e)
            )

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        import soundfile as sf

        # Load 5 s audio at 32 kHz
        y, sr = sf.read(audio_file_path, dtype="float32")
        if sr != self.sample_rate:
            # Use librosa for resampling if needed
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        target_len = int(self.sample_rate * self.window_size_s)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")

        # Convert waveform to Perch examples and embed
        examples = self._wav_to_examples(y, self.sample_rate)
        # examples: [num_windows, time, freq] -> embed to [num, 1280]
        emb = self._embed_model(examples)
        # Pool across time windows (mean)
        vec = np.mean(emb, axis=0).astype(np.float32)
        # L2 normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.from_numpy(vec).unsqueeze(0)


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    elif model_name == "vgg":
        model = models.vgg11(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, 2)

    elif model_name == "birdnet":
        # Return wrapped model for now
        model_path = "/workspaces/non-avian-ml/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
        return BirdNETModel(model_path), device

    elif model_name == "perch":
        # Return wrapped model for now
        return PerchModel(), device

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device), device
