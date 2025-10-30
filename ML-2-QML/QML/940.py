"""Quantum regression model with a parameter‑efficient variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Adds optional Gaussian noise to labels.
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
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset returning a dictionary with'states' (complex amplitudes) and 'target' (labels).
    """
    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Variational quantum circuit with a fixed entangling layer and a learnable encoder.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, n_ops: int = 20):
            super().__init__()
            self.n_wires = num_wires
            self.entangle = tq.CNOTList(cnot_list=[(i, (i+1)%num_wires) for i in range(num_wires)])
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.entangle(qdev)
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, n_ops: int = 20):
        super().__init__()
        self.n_wires = num_wires
        # Learnable encoder mapping classical features to quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, n_ops)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch shape: (batch, 2**num_wires)
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)
