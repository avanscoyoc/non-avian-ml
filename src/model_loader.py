import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, VGG11_Weights
import numpy as np
import librosa

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    tflite = None


class BirdNETModel:
    """BirdNET feature extractor using TFLite model logits as embeddings."""

    def __init__(self, model_path: str):
        if tflite is None:
            raise ImportError("ai_edge_litert not available")

        self.model_path = model_path
        self.sample_rate = 48000
        self.length_s = 3.0

        print(f"[BirdNET] Loading model: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.feature_dim = self.output_details[0]["shape"][1]
        print(f"[BirdNET] Model loaded (embedding dim: {self.feature_dim})")

    def _load_audio(self, path: str) -> np.ndarray:
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        target_len = int(self.sample_rate * self.length_s)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        return y.astype(np.float32)

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        audio = self._load_audio(audio_file_path)
        inp = np.expand_dims(audio, axis=0)

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(self.output_index)
        vec = logits[0].astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.from_numpy(vec).unsqueeze(0)


class PerchModel:
    """Perch feature extractor using TensorFlow Hub SavedModel."""

    def __init__(self, model_path: str):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        import tensorflow_hub as hub
        import soundfile as sf

        self.tf = tf
        self.sf = sf
        self.window_size_s = 5.0
        self.sample_rate = 32000
        self.feature_dim = 1280

        print(f"[Perch] Loading model: {model_path}")
        self._model = hub.load(model_path)
        if hasattr(self._model, 'infer_tf'):
            self._infer = self._model.infer_tf
        elif 'serving_default' in self._model.signatures:
            self._infer = self._model.signatures['serving_default']
        else:
            self._infer = self._model
        print(f"[Perch] Model loaded (embedding dim: {self.feature_dim})")

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        y, sr = self.sf.read(audio_file_path, dtype="float32")
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        target_len = int(self.sample_rate * self.window_size_s)
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start : start + target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")

        waveform = self.tf.constant(y[np.newaxis, :], dtype=self.tf.float32)

        if callable(self._infer):
            output = self._infer(waveform)
        else:
            output = self._infer

        if isinstance(output, dict):
            if 'embedding' in output:
                embeddings = output['embedding'].numpy()
            elif 'output_0' in output:
                embeddings = output['output_0'].numpy()
            else:
                embeddings = list(output.values())[0].numpy()
        else:
            embeddings = output.numpy()

        vec = embeddings[0].astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.from_numpy(vec).unsqueeze(0)


class CNNEmbeddingModel(nn.Module):
    """Frozen pretrained CNN feature extractor for audio spectrograms."""

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet":
            base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            base_model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512

        elif backbone_name == "mobilenet":
            base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            base_model.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.feature_extractor = base_model.features
            self.feature_dim = 1280
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone_name == "vgg":
            base_model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
            base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
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

        for param in self.parameters():
            param.requires_grad = False

        print(f"[{backbone_name.upper()}] Model loaded (embedding dim: {self.feature_dim})")

    def forward(self, x):
        x = self.feature_extractor(x)

        if self.backbone_name == "mobilenet":
            x = self.pool(x)
            x = torch.flatten(x, 1)
        elif self.backbone_name == "vgg":
            x = self.pool(x)
            x = self.flatten_fc(x)
        else:
            x = torch.flatten(x, 1)

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

    def extract_embeddings(self, audio_file_path: str) -> torch.Tensor:
        from data_loader import preprocess_audio

        mel_spec = preprocess_audio(audio_file_path)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.forward(mel_spec)

        return embeddings


def load_model(model_name: str):
    """Load pretrained model as frozen feature extractor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ["resnet", "mobilenet", "vgg"]:
        model = CNNEmbeddingModel(model_name)
        return model.to(device), device

    elif model_name == "birdnet":
        model_path = "/workspaces/non-avian-ml/model_birdnet_2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
        return BirdNETModel(model_path), device

    elif model_name == "perch":
        model_path = "/workspaces/non-avian-ml/model_perch_8"
        return PerchModel(model_path), device

    else:
        raise ValueError(f"Unknown model: {model_name}")
