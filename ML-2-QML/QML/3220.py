"""Hybrid quantum regression model that uses a learnable convolutional encoder followed by a variational quantum circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from Conv import Conv  # Import the classical convolutional filter


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a sinusoidal relationship in a quantum state."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class HybridRegressionDataset(torch.utils.data.Dataset):
    """Dataset that provides both a quantum state and a 2×2 image representation."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        # Convert the first 4 components of the state into a 2×2 image
        self.images = self.states[:, :4].real.reshape(-1, 1, 2, 2)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "state": torch.tensor(self.states[index], dtype=torch.cfloat),
            "image": torch.tensor(self.images[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridQuantumRegression(tq.QuantumModule):
    """Quantum regression model that encodes classical image data into a quantum circuit
    via a learnable convolutional layer, then applies a variational circuit and a linear head."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Classical convolutional encoder that outputs a scalar weight
        self.conv_filter = Conv()
        # Map the conv scalar to a rotation angle for the first qubit
        self.angle_mapper = nn.Linear(1, 1)
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode classical image via convolutional filter
        conv_score = self.conv_filter.run(batch["image"].squeeze(0).numpy())
        conv_tensor = torch.tensor(conv_score, dtype=torch.float32, device=batch["state"].device)
        # Use conv score to bias the first qubit's rotation
        angle = self.angle_mapper(conv_tensor.unsqueeze(0))
        # Prepare quantum device
        bsz = batch["state"].shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=batch["state"].device)
        # Encode the quantum state
        qdev.state = batch["state"]
        # Apply a rotation on the first qubit conditioned on the conv score
        qdev.rx(angle, wires=0)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure and feed into linear head
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionDataset", "HybridQuantumRegression", "generate_superposition_data"]
