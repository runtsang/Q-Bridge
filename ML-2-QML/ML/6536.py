import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Lightweight residual block used to deepen the feature extractor."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out += self.shortcut(x)
        return F.relu(out)

class HybridQuantumBinaryClassifier(nn.Module):
    """Classical backbone that can optionally feed a quantum head."""
    def __init__(self, num_classes: int = 2, use_quantum: bool = False,
                 quantum_head: nn.Module | None = None):
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_head = quantum_head

        # Residual CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Linear head producing 4 features for quantum encoding
        self.fc = nn.Linear(128, 4)
        # Classical classifier for the non‑quantum mode
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        if self.use_quantum and self.quantum_head is not None:
            logits = self.quantum_head(features)
        else:
            logits = self.classifier(features)
        return logits

__all__ = ["ResidualBlock", "HybridQuantumBinaryClassifier"]
