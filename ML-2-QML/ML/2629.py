import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFCModel(nn.Module):
    """
    Classical hybrid model that mimics a quantum fully connected layer.
    Architecture:
        CNN feature extractor -> fully connected projection -> classical quantum-inspired layer -> batch norm.
    The quantum-inspired layer uses a parameterized linear transformation followed by a tanh
    activation to emulate a quantum expectation value.
    """
    def __init__(self, n_classes: int = 4) -> None:
        super().__init__()
        # Convolutional feature extractor (mirrors QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        # Classical quantum-inspired layer
        self.qinspired = nn.Linear(n_classes, 1, bias=False)
        # Batch normalization on the final scalar output
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected projection
        x = self.fc(x)
        # Classical quantum-inspired expectation
        q_out = torch.tanh(self.qinspired(x))
        # Batch norm
        out = self.bn(q_out)
        return out

def FCL() -> HybridFCModel:
    """Return an instance of the hybrid classical model."""
    return HybridFCModel()

__all__ = ["HybridFCModel", "FCL"]
