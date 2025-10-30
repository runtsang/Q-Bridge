"""Quantum quanvolution regressor inspired by the quantum convolution filter and regression head.

This module implements a hybrid quantum-classical architecture that maps image patches to quantum states,
applies a random quantum circuit, measures all qubits, and uses a linear regression head to predict a scalar
output. The design borrows from the QuantumRegression module's encoder, random layers, and measurement.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QuanvolutionRegressor(tq.QuantumModule):
    """Quantum quanvolution filter followed by a regression head."""
    def __init__(self, patch_size: int = 2, stride: int = 2, out_channels: int = 4):
        super().__init__()
        self.n_wires = out_channels
        # Encoder that applies Ry to each of the 4 input pixels
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Regression head
        self.head = nn.Linear(self.n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on a batch of images.

        Args:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            Tensor of shape (batch,) containing regression outputs.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Stack pixels to shape (batch, 4)
                data = torch.stack(
                    [
                        x[:, 0, r, c],
                        x[:, 0, r, c + 1],
                        x[:, 0, r + 1, c],
                        x[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement)
        features = torch.cat(patches, dim=1)
        out = self.head(features)
        return out.squeeze(-1)

def generate_image_regression_data(num_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random 28×28 grayscale images and targets as the sum of pixel values."""
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.0, 1.0, size=(num_samples, 1, 28, 28)).astype(np.float32)
    targets = images.reshape(num_samples, -1).sum(axis=1).astype(np.float32)
    return images, targets

__all__ = ["QuanvolutionRegressor", "generate_image_regression_data"]
