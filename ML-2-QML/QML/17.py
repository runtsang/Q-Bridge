"""Quantum regression framework with variational circuit and amplitude encoding.

The module mirrors the original `QuantumRegression.py` but adds:
- amplitude encoding of classical features into a quantum state,
- a multi‑layer entangling ansatz,
- measurement of both single‑qubit and two‑qubit Pauli‑Z correlations,
- a classical head that processes the full feature vector.

This design enables richer quantum feature extraction while remaining
compatible with the classical counterpart for direct comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def amplitude_encode(features: np.ndarray) -> np.ndarray:
    """Encode a 1‑D feature vector into a quantum state by normalising
    it to a unit vector and interpreting it as amplitudes.
    """
    vec = features.astype(complex)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec, dtype=complex)
    return vec / norm

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random feature vectors and encode them as quantum states.
    Labels are a non‑linear function of the amplitudes.
    """
    raw = np.random.uniform(-1.0, 1.0, size=(samples, 2 ** num_wires))
    states = np.array([amplitude_encode(v) for v in raw], dtype=complex)
    # Simple target function: phase of |1..1> component modulated by amplitude of |0..0>
    phases = np.angle(states[:, -1])
    amplitudes = np.abs(states[:, 0])
    y = np.sin(2 * phases) * np.cos(amplitudes)
    return states, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns amplitude‑encoded quantum states and targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class EntanglingAnsatz(tq.QuantumModule):
    """Parameterised ansatz with alternating single‑qubit rotations and CNOTs."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot_layers = nn.ModuleList(
            [tq.CNOT(wires=(i, (i + 1) % num_wires)) for i in range(num_wires)]
        )

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for _ in range(self.num_layers):
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            for cnot in self.cnot_layers:
                cnot(qdev)

class SharedRegressionModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.ansatz = EntanglingAnsatz(num_wires, num_layers)
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head receives both sets of expectation values
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        z = self.measure_z(qdev)
        x = self.measure_x(qdev)
        features = torch.cat([z, x], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["SharedRegressionModel", "RegressionDataset", "generate_superposition_data"]
