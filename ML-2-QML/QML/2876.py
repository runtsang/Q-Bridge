"""Hybrid self‑attention + regression model – quantum implementation.

The quantum branch uses TorchQuantum to encode classical data into a
parameter‑dependent superposition, applies a variational self‑attention
circuit, measures all qubits, and maps the measurement statistics to a
regression output.  The interface mirrors the classical version so the
two models can be benchmarked side‑by‑side.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0⟩ + e^{iφ} sin(theta)|1…1⟩.
    Labels are a smooth function of theta and φ.
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
    return states, labels


class RegressionDataset(Dataset):
    """
    Torch dataset wrapping the quantum‑style superposition data.
    Returns a dict with ``states`` (complex) and ``target`` tensors.
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


class QuantumSelfAttention(tq.QuantumModule):
    """
    Variational circuit that implements a quantum self‑attention style block.
    Uses a RandomLayer for entanglement and a trainable single‑qubit rotation
    per wire to modulate the attention weights.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class HybridQuantumRegression(tq.QuantumModule):
    """
    Quantum hybrid model that mirrors the classical architecture:
    a data encoder, a quantum self‑attention block, measurement,
    and a classical linear head for regression.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.attention = QuantumSelfAttention(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.attention(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def run(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward to match the classical interface.
        """
        return self.forward(state_batch)


__all__ = [
    "HybridQuantumRegression",
    "RegressionDataset",
    "generate_superposition_data",
    "QuantumSelfAttention",
]
