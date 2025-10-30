"""Hybrid classical model combining QCNN and Quantum‑NAT architectures."""
import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """
    Classical neural network that merges the QCNN fully‑connected stack
    with the Quantum‑NAT convolutional backbone.

    Architecture:
        * Two 2‑D convolution blocks (Conv → ReLU → MaxPool)
          – inspired by Quantum‑NAT.
        * Fully‑connected stack mirroring the QCNN layers with
          residual‑style connections, batch‑norm and dropout.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected stack mimicking QCNN
        self.fc1 = nn.Sequential(nn.Linear(16 * 7 * 7, 128), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.head = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm = nn.BatchNorm1d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Residual‑style fully‑connected block
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.head(out)
        return torch.sigmoid(out)

__all__ = ["QCNNHybrid"]
