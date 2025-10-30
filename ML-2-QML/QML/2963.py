"""Hybrid quantum regression model integrating a general encoder, a random layer,
and a sampler‑style circuit for richer feature extraction.

The model mirrors the classical HybridRegressionModel but uses torchquantum
to simulate the quantum operations. It encodes the input state, applies a
random layer and RX/RY rotations, then runs a sampler circuit (Ry/CX) and
measures all qubits. The concatenated expectation values form the feature
vector fed to a linear head.

The design allows head‑to‑head comparison with the classical counterpart
while preserving quantum‑centric contributions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Dataset wrapping the quantum superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerLayer(tq.QuantumModule):
    """Parameterised sampler circuit inspired by the Qiskit SamplerQNN."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.ry_inputs = tq.RY(has_params=True, trainable=True)
        self.ry_weights = tq.RY(has_params=True, trainable=True)
        self.cx = tq.CX()

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for wire in range(self.num_wires):
            self.ry_inputs(qdev, wires=wire)
        self.cx(qdev, wires=[0, 1])
        for wire in range(self.num_wires):
            self.ry_weights(qdev, wires=wire)
        self.cx(qdev, wires=[0, 1])

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that combines a general encoder, a random layer,
    and a sampler‑style circuit."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical data to a superposition over |0..0> and |1..1>.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.sampler_layer = SamplerLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires * 2, 1)  # concatenated features from q_layer and sampler

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode input data
        self.encoder(qdev, state_batch)

        # Apply random layer + RX/RY rotations
        self.q_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

        # Measure to get first set of features
        features_q = self.measure(qdev)

        # Run sampler circuit on the same device state
        self.sampler_layer(qdev)

        # Measure again to get second set of features
        features_s = self.measure(qdev)

        # Concatenate features from both branches
        features = torch.cat([features_q, features_s], dim=-1)

        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "SamplerLayer"]
