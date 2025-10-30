"""
Hybrid feature extractor with quantum kernel, Bayesian head, and dropout for uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionNet(tq.QuantumModule):
    """
    Quantum implementation of the Quanvolution network. It replaces the classical
    convolution with a random quantum kernel applied to 2x2 image patches.
    """

    def __init__(self, n_wires: int = 4, dropout: float = 0.5) -> None:
        super().__init__()
        self.n_wires = n_wires

        # Encoder maps each pixel to a rotation about Y
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Random quantum circuit layer
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Dropout for Bayesian uncertainty
        self.dropout = nn.Dropout(dropout)

        # Linear head for classification
        self.linear = nn.Linear(self.n_wires * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the quantum filter, dropout, and linear head.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to 28x28 image
        x = x.view(bsz, 28, 28)
        patches = []

        # Iterate over 2x2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))

        # Concatenate all patch features
        features = torch.cat(patches, dim=1)

        # Apply dropout before the linear head
        features = self.dropout(features)

        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionNet"]
