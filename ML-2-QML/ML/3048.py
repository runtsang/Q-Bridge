"""Hybrid classical‑quantum estimator combining a quantum quanvolution filter with a fully‑connected regressor.

The model optionally uses a quantum kernel (via torchquantum) to transform input images into a high‑dimensional feature
space before feeding them into a classical neural network.  This mirrors the “Quanvolution” idea while staying
completely classical in the training loop, allowing seamless integration with PyTorch optimizers.

The architecture:
  - Input image (1×28×28)
  - Quantum quanvolution filter: 2×2 patches are encoded on 4 qubits, a random layer is applied, and Pauli‑Z
    measurements yield 4‑dimensional feature vectors per patch.
  - Flattened features are passed through a two‑layer fully‑connected head producing a scalar regression output.

This design preserves the strengths of both seeds: the expressive quantum kernel from the quanvolution example
and the simplicity of the EstimatorQNN regressor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule, GeneralEncoder, RandomLayer, MeasureAll, PauliZ

__all__ = ["HybridEstimatorQNN"]


class QuantumQuanvolutionFilter(QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = MeasureAll(PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature vector of shape (B, 4 * 14 * 14).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # reshape to patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, 0, r, c],
                        x[:, 0, r, c + 1],
                        x[:, 0, r + 1, c],
                        x[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridEstimatorQNN(nn.Module):
    """Hybrid regressor that optionally uses a quantum quanvolution filter."""

    def __init__(self, use_quantum: bool = True) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.feature_extractor = QuantumQuanvolutionFilter()
        else:
            # classical convolution as a fallback
            self.feature_extractor = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # After flattening, the feature dimension is 4 * 14 * 14 = 784
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Regression output of shape (B, 1).
        """
        if self.use_quantum:
            features = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
            features = features.view(x.size(0), -1)
        return self.head(features)
