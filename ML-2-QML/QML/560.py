"""Quantum regression dataset and model derived from ``new_run_regression.py`` with extensions.

The quantum part now consists of:
- a parameter‑shared variational layer that applies a sequence of CX and RY gates across all wires,
- a random layer that injects additional trainable noise,
- measurement of Pauli‑Z on each qubit,
- a lightweight classical head that maps the expectation values to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same as the original but preserved for compatibility."""
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
    """Hybrid quantum‑classical regression model with parameter‑shared variational layer."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            # Random layer with trainable parameters
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            # Parameter‑shared rotation layer: one RY per wire
            self.ry = tq.RY(has_params=True, trainable=True)
            # Parameter‑shared entangling layer: CX between consecutive wires
            self.cx = tq.CX()

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            # Entangling pattern: CX between wire i and i+1 (cyclic)
            for i in range(self.n_wires):
                self.cx(qdev, wires=[i, (i + 1) % self.n_wires])
            # Apply shared RY to each wire
            for wire in range(self.n_wires):
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps the input state to the quantum device
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head mapping expectation values to scalar
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
