# src/model.py
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 3, pretrained: bool = True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
