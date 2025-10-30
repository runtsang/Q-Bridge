"""Quantum regression model with a hybrid classical‑quantum feature extractor.

The original seed defined a fully quantum regression model.  This
extension adds a classical embedding of the input states that is
concatenated with the quantum measurement outcomes before the final
regression head.  The resulting model can be trained end‑to‑end and
provides a richer representation that leverages both quantum and classical
information.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – unchanged from the seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and targets as tensors."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum feature extractor
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Variational layer that applies a random circuit followed by trainable rotations."""
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

# --------------------------------------------------------------------------- #
# Hybrid quantum‑classical regression model
# --------------------------------------------------------------------------- #
class QuantumRegression__gen399(tq.QuantumModule):
    """Hybrid regression model that concatenates quantum measurement outcomes
    with a classical embedding of the input states before the final linear head.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps the input state vector into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Quantum variational layer
        self.q_layer = QLayer(num_wires)
        # Measure all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear embedding of the input state (treated as a vector)
        self.classical_embed = nn.Linear(2 ** num_wires, 16)
        # Final regression head that operates on the concatenated features
        self.head = nn.Linear(16 + num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # Quantum device that processes the batch
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical state into the quantum device
        self.encoder(qdev, state_batch)
        # Apply the variational layer
        self.q_layer(qdev)
        # Quantum measurement features
        qfeat = self.measure(qdev)  # shape: (bsz, n_wires)
        # Classical embedding of the raw input state
        cfeat = self.classical_embed(state_batch)  # shape: (bsz, 16)
        # Concatenate and regress
        features = torch.cat([cfeat, qfeat], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegression__gen399", "RegressionDataset", "generate_superposition_data"]
