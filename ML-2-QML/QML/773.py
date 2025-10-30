"""Quantum regression model with an entangling variational circuit.

The module extends the original seed by adding:
* a feature‑map encoder that uses a predefined Ry‑based encoding,
* a layered ansatz with CNOT entanglement,
* measurement of both Z and X Pauli operators,
* a learnable head that maps the concatenated expectation values to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The labels are a trigonometric function of the angles.
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
    """Dataset that returns a single sample as a dict."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class EntanglingAnsatz(tq.QuantumModule):
    """Layered ansatz with parameterized single‑qubit rotations and CNOT entanglement."""

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, num_wires, 3))
        # CNOT pattern: wire i -> i+1 mod n
        self.cnot_pattern = [(i, (i + 1) % num_wires) for i in range(num_wires)]

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                # RX(θ) – RY(φ) – RZ(λ)
                tq.RX(self.params[layer, wire, 0], has_params=False)(qdev, wire)
                tq.RY(self.params[layer, wire, 1], has_params=False)(qdev, wire)
                tq.RZ(self.params[layer, wire, 2], has_params=False)(qdev, wire)
            # Entangle
            for control, target in self.cnot_pattern:
                tq.CNOT()(qdev, control, target)

class QModel(tq.QuantumModule):
    """
    Hybrid quantum‑classical regression model.

    * `encoder` implements a Ry‑based feature map.
    * `ansatz` is a deep entangling circuit with trainable single‑qubit gates.
    * `measure` obtains expectation values of both Z and X on each qubit.
    * `head` maps the concatenated expectation vector to a scalar output.
    """

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.ansatz = EntanglingAnsatz(num_wires, n_layers)
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        z_exp = self.measure_z(qdev)
        x_exp = self.measure_x(qdev)
        features = torch.cat([z_exp, x_exp], dim=-1)
        return self.head(features).squeeze(-1)
