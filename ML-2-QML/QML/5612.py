"""Quantum‑only regression model that preserves the original API but
adds a variational layer and a configurable entanglement scheme.

The circuit follows the same data‑uploading encoding as the seed,
but the head contains a RandomLayer + RX/RZ rotations and a
measurement of all qubits.  The model can be used both as a stand‑alone
quantum module and as a sub‑module of a hybrid network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and labels as in the seed."""
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

class RegressionDataset(Dataset):
    """Dataset that yields a dictionary with ``states`` and ``target``."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionHybrid(tq.QuantumModule):
    """A quantum‑only regression model that mirrors the original
    ``QuantumRegression`` but adds a random layer and optional
    entanglement.  The model is fully differentiable via TorchQuantum.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, entangle: bool = True):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.layer = self._QLayer(num_wires)
        self.entangle = entangle
        if entangle:
            self.cnot = tq.CNOT
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (B, 2**num_wires) containing complex amplitudes
            ready for state preparation.

        Returns
        -------
        torch.Tensor
            Regression output of shape (B,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.layer(qdev)
        if self.entangle:
            for w in range(self.n_wires - 1):
                self.cnot(qdev, wires=[w, w + 1])
            self.cnot(qdev, wires=[self.n_wires - 1, 0])
        out = self.measure(qdev)
        return self.head(out).squeeze(-1)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
