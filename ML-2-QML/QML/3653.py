"""Quantum autoencoder with a regression head using torchquantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition |0..0> + e^{iϕ}|1..1> with sinusoidal labels."""
    omega0 = np.zeros(2 ** num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2 ** num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning complex quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumAutoencoderRegression(tq.QuantumModule):
    """Quantum autoencoder followed by a regression head."""
    class QEncoder(tq.QuantumModule):
        """Random parameterized layer for feature extraction."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.rand = tq.RandomLayer(n_ops=25, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.rand(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    class QDecoder(tq.QuantumModule):
        """Swap‑test based decoder to reconstruct the input."""
        def __init__(self, n_wires: int, trash_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.trash = trash_wires
            self.swap = tq.CSWAP()
            self.h = tq.Hadamard()

        def forward(self, qdev: tq.QuantumDevice):
            # Apply swap test between each input wire and a trash wire
            for i in range(self.n_wires):
                qdev.apply(self.h, wires=i)
                qdev.apply(self.swap, wires=[i, self.n_wires + i])
                qdev.apply(self.h, wires=i)

    def __init__(self, num_wires: int, trash_wires: int = 2):
        super().__init__()
        self.encoder = self.QEncoder(num_wires)
        self.decoder = self.QDecoder(num_wires, trash_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires + self.decoder.trash,
                                bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.decoder(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "QuantumAutoencoderRegression",
    "RegressionDataset",
    "generate_superposition_data",
]
