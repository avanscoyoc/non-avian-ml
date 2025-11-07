import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import librosa
import ai_edge_litert.interpreter as tflite


class BirdNETModel:
    """BirdNET feature extractor using logits as embeddings.

    - Input: 3 s audio at 48 kHz (shape [1, 144000])
    - Embeddings: 6522-d logit vector (before softmax)

    Note: We use logits instead of intermediate pooling layers because
    TFLite doesn't retain intermediate tensors by default.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sample_rate = 48000
        self.length_s = 3.0
        self.interpreter = None
        self.input_index = None
        self.output_index = None

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # I/O details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        # Feature dimension is the logit size
        self.feature_dim = self.output_details[0]["shape"][1]

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

        # Get logits (output before softmax)
        logits = self.interpreter.get_tensor(self.output_index)
        vec = logits[0].astype(np.float32)

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


class CNNEmbeddingModel(nn.Module):
    """Wrapper for torchvision CNNs that extracts frozen pretrained embeddings.

    Matches BirdNET/Perch workflow: freeze backbone, train only a new classifier head.
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet":
            base_model = models.resnet18(pretrained=True)
            # Adapt first conv for single-channel mel-spectrograms
            base_model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Remove final FC layer to extract embeddings
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512

        elif backbone_name == "mobilenet":
            base_model = models.mobilenet_v2(pretrained=True)
            # Adapt first conv
            base_model.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            # Remove classifier to extract embeddings
            self.feature_extractor = base_model.features
            self.feature_dim = 1280
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone_name == "vgg":
            base_model = models.vgg11(pretrained=True)
            # Adapt first conv
            base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            # Remove classifier to extract embeddings
            self.feature_extractor = base_model.features
            self.feature_dim = 512
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.flatten_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Freeze all pretrained weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract frozen embeddings from mel-spectrogram input."""
        x = self.feature_extractor(x)

        if self.backbone_name == "mobilenet":
            x = self.pool(x)
            x = torch.flatten(x, 1)
        elif self.backbone_name == "vgg":
            x = self.pool(x)
            x = self.flatten_fc(x)
        else:  # resnet
            x = torch.flatten(x, 1)

        # L2 normalize
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        """Extract embeddings from audio file (for consistency with BirdNET/Perch API)."""
        from data_loader import preprocess_audio

        # Preprocess audio to mel-spectrogram
        mel_spec = preprocess_audio(audio_file_path)

        # Add batch dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.forward(mel_spec)

        return embeddings


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ["resnet", "mobilenet", "vgg"]:
        # Return frozen embedding extractor
        model = CNNEmbeddingModel(model_name)
        return model.to(device), device

    elif model_name == "birdnet":
        # Return wrapped model for now
        model_path = "/workspaces/non-avian-ml/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
        return BirdNETModel(model_path), device

    elif model_name == "perch":
        # Return wrapped model for now
        return PerchModel(), device

    else:
        raise ValueError(f"Unknown model: {model_name}")
