"""Hybrid quantum regression model that uses a quanvolutional filter
followed by a linear head.  The class names mirror the classical
counterpart to enable drop‑in replacement in downstream pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset.

    The same data generation routine as the classical version, but
    the returned states are kept as real tensors because the encoder
    only needs real amplitudes.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    H = W = int(np.sqrt(num_features))
    x = x.reshape(samples, H, W)
    angles = x.sum(axis=(1, 2))
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset compatible with the quantum model."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuanvolutionFilter(tq.QuantumModule):
    """
    Variational quantum kernel that operates on 2×2 image patches.

    The filter encodes each pixel with a Ry gate, applies a random
    two‑qubit circuit, and measures all qubits in the Z basis.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum kernel to all 2×2 patches of the input batch.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, H, W)``.

        Returns
        -------
        torch.Tensor
            Concatenated measurement features of shape
            ``(batch, num_patches)``.
        """
        bsz, H, W = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(H - 1):
            for c in range(W - 1):
                patch = x[:, r : r + 2, c : c + 2].reshape(bsz, 4)
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement)
        return torch.cat(patches, dim=1)


class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical
    ``HybridRegressionModel``.  It uses the ``QuanvolutionFilter``
    as a feature extractor and a linear head for regression.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        H = int(np.sqrt(num_features))
        self.num_patches = (H - 1) * (H - 1)
        self.qfilter = QuanvolutionFilter()
        self.head = nn.Linear(self.num_patches, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape ``(batch, H, W)``.

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape ``(batch,)``.
        """
        features = self.qfilter(state_batch)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
