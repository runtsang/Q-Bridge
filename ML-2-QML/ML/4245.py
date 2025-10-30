"""Hybrid Quantum–Classical Natural Language Model (ML side)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum module defined in the QML file
from.QuantumNAT_QML import QFCQuantum  # relative import to the QML implementation


class SamplerQNN(nn.Module):
    """
    Classical soft‑max sampler that maps the 2‑dimensional quantum output
    to a probability distribution.  It mirrors the structure of the
    quantum SamplerQNN but remains entirely classical for speed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class QuantumNATHybrid(nn.Module):
    """
    Hybrid CNN → Quantum Expectation → Sampler pipeline.

    * Convolutional backbone extracts image features.
    * Fully‑connected layers reduce dimensionality to 2 features.
    * QFCQuantum maps the 2‑dimensional vector to quantum expectation values.
    * SamplerQNN produces calibrated probabilities for downstream tasks.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Fully‑connected head
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 outputs for the quantum encoder

        # Quantum and sampling sub‑modules
        self.quantum = QFCQuantum()
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten and fully‑connected transformation
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Shape: (batch, 2)

        # Quantum expectation head
        x = self.quantum(x)  # Shape: (batch, 2)

        # Sampler to produce probabilities
        x = self.sampler(x)  # Shape: (batch, 2)

        return x


__all__ = ["QuantumNATHybrid", "SamplerQNN"]
