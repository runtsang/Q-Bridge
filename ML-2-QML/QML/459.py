"""Quantum regression model with a depth‑controlled variational ansatz and learnable encoding.

The module builds upon the original seed by:
- Introducing a configurable depth for the QLayer.
- Adding a trainable GeneralEncoder per wire (parameter‑shift enabled).
- Using a measurement that returns a feature vector of shape (batch, n_wires).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of states of the form
    cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
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
    Dataset that returns a dictionary with keys ``states`` and ``target``.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Quantum regression model with a depth‑controlled variational circuit.
    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    depth : int, optional
        Depth of the variational layer. Defaults to 3.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 3):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=20 * depth, wires=list(range(num_wires)))
            # Trainable rotation gates for each wire
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Apply a random layer followed by a depth‑controlled
            # sequence of single‑qubit rotations.
            self.random_layer(qdev)
            for _ in range(self.depth):
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = num_wires
        # Learnable encoding: a trainable RX gate per wire
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRX"]
        )
        self.q_layer = self.QLayer(num_wires, depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)
