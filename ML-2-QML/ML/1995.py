import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """
    Classical hybrid model extending the original QFCModel with a residual CNN backbone,
    dropout, and layer‑norm for improved generalization. The architecture mirrors the
    seed but adds a skip connection after the second pooling layer.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        # Base feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Residual block
        self.residual = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        self.residual_bn = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(dropout)
        # Fully connected head
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        residual = self.residual(features)
        features = features + self.residual_bn(residual)
        features = F.relu(features, inplace=True)
        flattened = features.view(features.size(0), -1)
        out = self.classifier(flattened)
        return self.norm(out)
