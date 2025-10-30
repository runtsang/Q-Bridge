"""Hybrid fully‑connected layer with a classical CNN backbone and a quantum‑style feature map.

The module combines the convolutional feature extraction of the Quantum‑NAT example with a
parameterised “quantum‑inspired’’ linear transformation that mimics a single‑qubit
rotational circuit.  Training remains fully classical while the quantum motif
provides richer non‑linear embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumFeatureMap(nn.Module):
    """Classical approximation to a single‑qubit rotation‑based expectation value.

    The module implements ``E(θ) = tanh(w·θ + b)``, which is equivalent to the
    expectation of Pauli‑Z after a Ry rotation with trainable parameters.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class HybridFCL(nn.Module):
    """CNN → FC → quantum‑inspired feature map → output.

    The architecture mirrors the QFCModel of Quantum‑NAT but replaces the
    quantum block with a differentiable feature map that can be trained
    end‑to‑end with standard optimisers.
    """
    def __init__(self, out_features: int = 4):
        super().__init__()
        # Feature extractor (identical to QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        # Quantum‑inspired feature map applied element‑wise
        self.qmap = QuantumFeatureMap(out_features)

        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        feat = feat.view(bsz, -1)
        out = self.fc(feat)
        # Apply quantum‑style transformation to each output channel
        out = self.qmap(out)
        return self.norm(out)
