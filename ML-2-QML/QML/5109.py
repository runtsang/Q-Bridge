"""Hybrid quantum‑classical regression model using TorchQuantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a quantum circuit.  The state is a superposition of
    |0…0> and |1…1> with random angles θ, φ.
    The target value is sin(2θ) * cos(φ) and we keep the label
    to be stable across both models.
    """
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
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary containing the quantum state and target.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionEnhanced(tq.QuantumModule):
    """
    Hybrid quantum‑classical regression model using a variational circuit.
    """
    class QLayer(tq.QuantumModule):
        """
        Variational layer that mixes the encoded state with trainable rotations.
        """
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
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
        # Encode each amplitude into a rotation on its corresponding qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(num_wires)
            ]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: Tensor of shape (batch, 2**num_wires) with dtype complex.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionEnhanced", "RegressionDataset", "generate_superposition_data"]
