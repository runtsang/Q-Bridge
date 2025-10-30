import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATGen065(nn.Module):
    """Hybrid classical regression model that extends the original QuantumNAT architecture.
    Combines a 2‑D convolutional backbone, a dense regression head, and optional
    estimator utilities for noisy evaluation.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional encoder
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        # Dense regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        out = self.head(features)
        return self.norm(out).squeeze(-1)

__all__ = ["QuantumNATGen065"]
