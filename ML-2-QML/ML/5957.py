import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    """Feature extractor with a multi‑stage convolutional pipeline."""
    def __init__(self, in_channels: int = 3, base_f: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_f, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(base_f, base_f * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

class HybridClassifier(nn.Module):
    """Purely classical binary classifier that mirrors the hybrid design."""
    def __init__(self, in_channels: int = 3, base_f: int = 32):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, base_f)
        self.fc1 = nn.Linear(base_f * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridClassifier"]
