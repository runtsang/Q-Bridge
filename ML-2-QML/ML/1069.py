"""HybridQuantumCNN – classical backbone + quantum head for binary classification.

This module implements a deep CNN that feeds a 3‑dimensional feature vector into a
parameterised 3‑qubit quantum circuit.  The quantum circuit outputs a single
expectation value which is interpreted as the probability of class 1.  The
classical network therefore produces a 2‑logit soft‑max output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum head from the QML module.
# The QML module must be in the same package or Python path.
import quantum_module  # type: ignore

class HybridQuantumCNN(nn.Module):
    """CNN backbone followed by a 3‑qubit quantum expectation head."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.3) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
        )

        # Reduce the spatial dimensions to a flat vector
        self.flatten = nn.Flatten()

        # Linear layer to compress the feature vector to 3 angles
        self.angle_mapper = nn.Linear(64 * 4 * 4, 3)  # assuming input 32x32

        # Quantum head
        self.quantum_head = quantum_module.HybridQuantumCNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        x = self.flatten(x)
        # Map to 3 angles
        angles = self.angle_mapper(x)
        # Quantum expectation
        prob = self.quantum_head(angles)  # shape (batch, 1)
        # Convert to logits for two classes
        logits = torch.cat([prob, 1 - prob], dim=-1)
        return logits

__all__ = ["HybridQuantumCNN"]
