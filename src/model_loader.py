import os
import torch
from torchvision import models
import torch.nn as nn

class ModelLoader:
    def __init__(self, model_name, num_classes=2):
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.device = self._get_device()
        self.model = None

    def get_model(self):
        if self.model is None:
            if self.model_name == 'resnet':
                self.model = self._get_resnet()
            elif self.model_name == 'mobilenet':
                self.model = self._get_mobilenet()
            elif self.model_name == 'vgg':
                self.model = self._get_vgg()
            elif self.model_name in ['birdnet', 'perch']:
                self.model = self._get_bioacoustics_model()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        return self.model

    def _get_resnet(self):
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model.to(self.device)
    
    def _get_mobilenet(self):
        model = models.mobilenet_v2(pretrained=True)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model.to(self.device)

    def _get_vgg(self):
        model = models.vgg11(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, self.num_classes)
        return model.to(self.device)

    def _get_bioacoustics_model(self):
        # Convert lowercase model names to correct case
        model_name = "BirdNET" if self.model_name == "birdnet" else "Perch"
        model = torch.hub.load(
            "kitzeslab/bioacoustics-model-zoo", model_name, trust_repo=True
        )
        return model.to(self.device)

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")