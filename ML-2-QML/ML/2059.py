import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Extended CNN‑MLP with an embedding layer and depth‑wise separable convolutions."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        # Embedding: convert 28x28 grayscale image into 16‑channel feature map
        self.embedding = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        # Depth‑wise separable conv block
        self.dw_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)
        # Residual block
        self.res_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
        )
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.embedding(x)
        features = self.dw_conv(features)
        features = self.pool(features)
        residual = features
        features = self.res_block(features)
        features += residual
        features = self.pool(features)
        flattened = features.view(features.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)

__all__ = ["QFCModel"]
