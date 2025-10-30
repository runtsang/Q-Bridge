"""Hybrid quantum regression model that mirrors the classical autoencoder structure.

The quantum circuit first encodes the classical feature vector into a
superposition using a feature map.  It then applies a random layer
followed by a parameter‑dependent rotation on each qubit.  The expectation
values of Pauli‑Z are measured and fed into a classical linear head.
This design is inspired by the autoencoder circuit in the reference
and allows a side‑by‑side comparison with the classical model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumModule, QuantumDevice


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic quantum state dataset.

    Parameters
    ----------
    num_wires: int
        Number of qubits used in the circuit.
    samples: int
        Number of samples to generate.

    Returns
    -------
    states: np.ndarray of shape (samples, 2**num_wires)
        Complex amplitudes of the state |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    labels: np.ndarray of shape (samples,)
        Target values computed as ``sin(2θ) * cos(φ)``.
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
    """Dataset wrapper around the quantum state generation."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QLayer(tq.QuantumModule):
    """Variational layer with a random circuit followed by per‑wire rotations."""

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


class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that follows the classical autoencoder pipeline.

    The model consists of a feature‑map encoder, a variational layer,
    a measurement, and a linear head.  It can be trained end‑to‑end
    using a gradient‑based optimizer on the classical parameters of the
    rotations and the head.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Feature map – a simple Ry‑based encoding
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# Backward‑compatibility alias used in the original anchor
QModel = HybridRegressionModel

__all__ = [
    "HybridRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "QLayer",
    "QModel",
]
