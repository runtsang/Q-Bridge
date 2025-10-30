"""Quantum hybrid model combining QuantumNAT encoder with regression head, inspired by QuantumRegression."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATRegressionModel(tq.QuantumModule):
    """
    Quantum version of the hybrid model. It encodes input images or superposition states,
    applies a random variational layer, measures all qubits, and maps the expectation values
    to a scalar regression output or a 4‑class classification vector.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            self.crx(qdev, wires=[0, 1])
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 3], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4, task: str = "classification") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.task = task
        # Encoder that maps classical features to qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 4 if task == "classification" else 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input (image patches or superposition states)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex superposition states for quantum regression, mirroring QuantumRegression.
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
    """
    Dataset that returns quantum states and regression targets.
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

__all__ = ["QuantumNATRegressionModel", "RegressionDataset", "generate_superposition_data"]
