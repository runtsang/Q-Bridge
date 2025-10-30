"""Quantum regression model with a parameter‑shared variational circuit and classical head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


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
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum state vectors and target values."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class VariationalCircuit(tq.QuantumModule):
    """Parameter‑shared variational circuit with entanglement and rotation layers."""

    def __init__(self, n_wires: int, n_layers: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # Entanglement pattern (CNOT between consecutive wires)
            self.layers.append(tq.EntanglementLayer(n_wires, entanglement="cnot"))
            # Trainable rotation gates
            self.layers.append(tq.RX(has_params=True, trainable=True))
            self.layers.append(tq.RY(has_params=True, trainable=True))

    def forward(self, qdev: tq.QuantumDevice):
        for layer in self.layers:
            layer(qdev)


class QuantumRegression(tq.QuantumModule):
    """Hybrid quantum‑classical regression model."""

    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.n_wires = num_wires
        # Learnable encoding of input state vectors
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational circuit
        self.var_circuit = VariationalCircuit(num_wires, n_layers)
        # Expectation value readout
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state vectors
        self.encoder(qdev, state_batch)
        # Apply the variational circuit
        self.var_circuit(qdev)
        # Readout features
        features = self.measure(qdev)
        # Classical linear head
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
