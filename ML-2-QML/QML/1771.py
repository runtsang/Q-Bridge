"""Quantum regression model with depth‑controlled variational ansatz and hybrid measurement."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form |ψ⟩ = cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩.
    The labels are a smooth function of θ and ϕ.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i, (t, p) in enumerate(zip(thetas, phis)):
        states[i] = np.cos(t) * omega_0 + np.exp(1j * p) * np.sin(t) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

# ------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------
class RegressionDataset(torch.utils.data.Dataset):
    """
    Returns a dictionary with complex state vectors and real targets.
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

# ------------------------------------------------------------
# Variational ansatz
# ------------------------------------------------------------
class VarDepthLayer(tq.QuantumModule):
    """
    A depth‑controlled variational block that applies RZ‑RX‑RY layers
    on every qubit for a specified depth.
    """
    def __init__(self, num_wires: int, depth: int):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        for d in range(self.depth):
            for w in range(self.num_wires):
                self.rz(qdev, wires=w)
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

# ------------------------------------------------------------
# Hybrid quantum‑classical model
# ------------------------------------------------------------
class QModel(tq.QuantumModule):
    """
    Quantum regression model using a depth‑controlled variational ansatz
    followed by a classical linear head.
    """
    def __init__(self, num_wires: int, depth: int = 1):
        super().__init__()
        self.n_wires = num_wires
        self.depth = depth
        # Simple angle‑based encoding
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        # Random layer for additional expressivity
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        # Variational depth layer
        self.q_layer = VarDepthLayer(num_wires, depth)
        # Measure all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head to produce a scalar output
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state vectors
        self.encoder(qdev, state_batch)
        # Apply randomization and variational layers
        self.random(qdev)
        self.q_layer(qdev)
        # Measure qubits to obtain classical features
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
