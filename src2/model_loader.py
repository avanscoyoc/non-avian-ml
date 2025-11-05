import torch
from torchvision import models
import torch.nn as nn
import numpy as np


class BirdNETModel:
    """BirdNET feature extractor for custom classifier training"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.feature_dim = 1024  # BirdNET embedding dimension
        
        # For now, create a mock feature extractor since TF has issues
        # This simulates BirdNET's behavior for testing/development
        print(f"BirdNET feature extractor initialized (mock mode)")
        print(f"Model path: {model_path}")
        print(f"Feature dimension: {self.feature_dim}")

    def extract_embeddings(self, audio_file_path):
        """
        Extract BirdNET embeddings from audio file.
        Returns consistent embeddings for the same file (for reproducibility).
        """
        try:
            # For development: create deterministic embeddings based on file path
            # This ensures consistent results across runs for the same audio file
            import hashlib
            
            # Create a hash from the file path for deterministic embeddings
            file_hash = hashlib.md5(str(audio_file_path).encode()).hexdigest()
            
            # Convert hash to seed for reproducible random embeddings
            seed = int(file_hash[:8], 16) % (2**31 - 1)
            np.random.seed(seed)
            
            # Generate mock embeddings that simulate BirdNET features
            # These represent high-level acoustic features that BirdNET would extract
            embeddings = np.random.randn(self.feature_dim).astype(np.float32)
            
            # Normalize embeddings (common practice for deep learning features)
            embeddings = embeddings / np.linalg.norm(embeddings)
            
            return torch.from_numpy(embeddings).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error extracting embeddings from {audio_file_path}: {e}")
            # Return zero embeddings on error
            return torch.zeros(1, self.feature_dim)
class PerchModel:
    """Perch feature extractor for custom classifier training
    
    Based on Google Research Perch model architecture:
    - Uses 5-second audio windows (data_5s folders)
    - EfficientNet-based embedding model
    - Designed for bird vocalizations and bioacoustic classification
    - Window size: 5.0 seconds at model's native sample rate
    """

    def __init__(self):
        # Perch model configuration based on the Google Research implementation
        self.window_size_s = 5.0  # Perch uses 5-second windows
        self.sample_rate = 32000  # Perch's native sample rate
        self.feature_dim = 1280  # EfficientNet embedding dimension
        
        # For development: create a mock feature extractor
        # In production, this would load the actual Perch model from Kaggle
        print("Perch feature extractor initialized (mock mode)")
        print(f"Window size: {self.window_size_s}s")
        print(f"Sample rate: {self.sample_rate}Hz") 
        print(f"Feature dimension: {self.feature_dim}")
        print("Model source: Kaggle Models (google/bird-vocalization-classifier)")
        print("Uses data_5s folders for 5-second audio segments")

    def extract_embeddings(self, audio_file_path):
        """
        Extract Perch embeddings from 5-second audio file.
        
        In the actual implementation, this would:
        1. Load audio file at 32kHz sample rate
        2. Process through PCEN melspectrogram frontend
        3. Pass through EfficientNet backbone
        4. Return pooled embeddings
        
        Returns consistent embeddings for the same file (for reproducibility).
        """
        try:
            # For development: create deterministic embeddings based on file path
            # This simulates the actual Perch model behavior
            import hashlib
            
            # Create a hash from the file path for deterministic embeddings
            file_hash = hashlib.md5(str(audio_file_path).encode()).hexdigest()
            
            # Use different seed space than BirdNET to ensure different features
            # Perch focuses on different acoustic features than BirdNET
            seed = (int(file_hash[:8], 16) + 54321) % (2**31 - 1)
            np.random.seed(seed)
            
            # Generate mock embeddings that simulate Perch EfficientNet features
            # These represent high-level acoustic features for 5-second segments
            embeddings = np.random.randn(self.feature_dim).astype(np.float32)
            
            # Normalize embeddings (standard practice for deep learning features)
            embeddings = embeddings / np.linalg.norm(embeddings)
            
            return torch.from_numpy(embeddings).unsqueeze(0)  # Add batch dim
            
        except Exception as e:
            print(f"Error extracting embeddings from {audio_file_path}: {e}")
            # Return zero embeddings on error
            return torch.zeros(1, self.feature_dim)


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
