import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Classical branch of the enhanced Quantum‑NAT model.
    Combines a deeper CNN, dropout, and a learnable embedding before the final projection.
    """
    def __init__(self):
        super().__init__()
        # Feature extractor: 3 conv blocks with batchnorm and dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Embedding layer with dropout
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
        )
        # Final projection to 4 outputs
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        embedded = self.embed(features)
        out = self.fc(embedded)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
