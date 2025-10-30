"""Quantum regression model with a variational circuit and controlled‑phase layer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics a superposition state.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        States array of shape (samples, 2**num_wires) and target array of shape (samples,).
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

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and targets."""

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
    """Quantum regression model with a parameter‑shared variational circuit and a
    controlled‑phase layer to capture higher‑order correlations."""

    class QLayer(tq.QuantumModule):
        """Variational layer with shared parameters and a controlled‑phase gate."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random layer for initial mixing
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            # Parameter‑shared RX and RY rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            # Controlled‑phase gate between adjacent wires
            self.cz = tq.CZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            # Apply random mixing
            self.random_layer(qdev)
            # Apply shared rotations
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Apply controlled‑phase between neighboring wires
            for wire in range(self.n_wires - 1):
                self.cz(qdev, wires=(wire, wire + 1))

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encode input states using a simple Ry rotation per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
