"""Classical baseline model that emulates the quantum architecture using
classical convolutions and a random 2×2 filter.

The network mirrors the feature extraction stages of the quantum version
and replaces the variational circuit with a deterministic linear layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNAT(nn.Module):
    """Classical baseline that mimics the quantum architecture:
    a CNN feature extractor, a random 2×2 convolution (classical quanvolution),
    and a fully connected head."""
    def __init__(self) -> None:
        super().__init__()
        # Classical CNN extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quanvolution: 2×2 conv with random weights
        self.quanvolution = nn.Conv2d(16, 4, kernel_size=2, stride=1, bias=False)
        nn.init.normal_(self.quanvolution.weight, mean=0.0, std=0.1)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(4 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = self.cnn(x)
        x = self.quanvolution(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["HybridQuantumNAT"]
