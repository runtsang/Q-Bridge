from __future__ import annotations

import torch
from torch import nn

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid classical neural network that emulates a QCNN architecture
    while retaining the lightweight feed‑forward structure of the
    original EstimatorQNN.  The model first maps the 2‑dimensional
    input into an 8‑dimensional representation, then applies a
    series of convolutional‑style linear layers and pooling stages
    inspired by the QCNN reference.  The final linear head produces
    a scalar output suitable for regression or binary classification.
    The network exposes ``get_quantum_weights`` to return a flat
    tensor that can be supplied to a Qiskit EstimatorQNN as the
    weight parameters of the quantum ansatz.
    """
    def __init__(self) -> None:
        super().__init__()
        # Initial embedder
        self.embed = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 8),
        )
        # QCNN‑style layers
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.embed(inputs)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def get_quantum_weights(self) -> torch.Tensor:
        """
        Flatten all parameters that will be mapped to the quantum
        ansatz weight parameters.  The order follows the sequence
        of layers: conv1, pool1, conv2, pool2, conv3.
        """
        params = []
        for layer in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3]:
            for p in layer.parameters():
                params.append(p.view(-1))
        return torch.cat(params, dim=0)

__all__ = ["HybridEstimatorQNN"]
