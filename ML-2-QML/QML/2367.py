"""Quantum hybrid model combining a quantum quanvolution filter with a quantum regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


class HybridQuanvolutionRegression(tq.QuantumModule):
    """Quantum hybrid model combining a quantum quanvolution filter with a quantum regression head."""

    class QLayer(tq.QuantumModule):
        """Inner quantum layer applying random ops and trainable rotations."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(num_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int = 4,
        patch_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.patch_size = patch_size
        self.stride = stride
        # Encoder mapping classical pixel values to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        bsz = x.shape[0]
        device = x.device
        # Flatten to (batch, height, width)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev = tq.QuantumDevice(
                    self.num_wires, bsz=bsz, device=device
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.num_wires))
        # Concatenate all patch measurements
        features = torch.cat(patches, dim=1)
        return self.head(features).squeeze(-1)


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states for regression tasks."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


__all__ = ["HybridQuanvolutionRegression", "RegressionDataset", "generate_superposition_data"]
