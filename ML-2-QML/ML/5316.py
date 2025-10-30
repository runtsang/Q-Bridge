import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class UnifiedHybridLayer(nn.Module):
    """
    Classical hybrid layer that can emulate the behaviour of the original
    FCL and QFCModel classes while exposing a single forward interface.
    The module consists of a classical feature extractor (CNN or linear)
    followed by a differentiable head that mimics a quantum expectation
    using a linear layer and sigmoid activation.
    """

    def __init__(self, in_channels: int = 1, mode: str = "cnn"):
        super().__init__()
        if mode == "cnn":
            # CNN backbone inspired by QuantumNAT
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.norm = nn.BatchNorm1d(1)
        elif mode == "fcl":
            # Fully‑connected layer inspired by FCL
            self.linear = nn.Linear(1, 1)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        # Classical head mimicking a quantum expectation
        self.head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "features"):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = self.norm(out)
        else:
            out = self.linear(x)
        # Apply the classical head
        return self.head(out)
