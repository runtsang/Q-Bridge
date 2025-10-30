"""Hybrid quantum regression model that uses a random layer and
parameterised rotations, followed by a classical linear head.
The implementation is a direct extension of the original
QuantumRegression example, incorporating a fully‑connected
quantum layer that mirrors the behaviour of the classical
QuantumInspiredLayer defined in the ML counterpart."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate identical state vectors and labels to the classical
    counterpart.  Each state is a superposition of |0..0⟩ and |1..1⟩
    with random amplitudes."""
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

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset that supplies complex state vectors and real labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class FullyConnectedQuantumLayer(tq.QuantumModule):
    """A quantum circuit that implements a fully‑connected quantum layer.
    It applies a random layer followed by a trainable Ry rotation on each
    qubit, then measures all qubits in the Pauli‑Z basis."""
    def __init__(self, num_wires: int, n_random_ops: int = 30):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=n_random_ops, wires=list(range(num_wires)))
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.ry(qdev, wires=wire)

class QuantumRegressionModel(tq.QuantumModule):
    """Full quantum regression model that encodes the input state,
    applies the fully‑connected quantum layer, measures all qubits,
    and uses a classical linear head to produce the scalar output."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = FullyConnectedQuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "QuantumRegressionDataset",
    "QuantumRegressionModel",
    "generate_superposition_data",
]
