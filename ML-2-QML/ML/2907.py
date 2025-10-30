"""Classical hybrid binary classifier with quantum-inspired expectation head.

This module implements a CNN backbone followed by a dense layer that produces two
parameters (θ, φ).  The expectation of a 1‑qubit circuit with H‑Ry‑Rx gates is
computed analytically as sin(θ)·sin(φ).  The result is passed through a
sigmoid to obtain class probabilities.  The implementation is fully
PyTorch‑based and uses no quantum libraries, enabling fast simulation and
gradient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumExpectation(torch.autograd.Function):
    """Analytical expectation of Y for a 1‑qubit H‑Ry‑Rx circuit."""
    @staticmethod
    def forward(ctx, params: torch.Tensor) -> torch.Tensor:
        # params shape: (batch, 2) -> [theta, phi]
        theta = params[..., 0]
        phi = params[..., 1]
        # Expectation of Y = sin(theta) * sin(phi)
        return torch.sin(theta) * torch.sin(phi)

class HybridBinaryClassifier(nn.Module):
    """CNN + quantum‑inspired head for binary classification."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Simple 2‑layer CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        # For 32×32 input images the flattened feature size is 15×6×6 = 540
        self.fc1 = nn.Linear(540, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # outputs [theta, phi]
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        params = self.fc3(x)  # (batch, 2)
        expectation = QuantumExpectation.apply(params)  # (batch,)
        logits = expectation + self.shift
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)
