"""Quantum regression model with amplitude encoding and a hardware‑efficient ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_quantum_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and sinusoidal labels."""
    thetas = np.random.uniform(0, np.pi, size=samples)
    phis = np.random.uniform(0, 2*np.pi, size=samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        # amplitude encoding: cos(theta)|0> + sin(theta) e^{i phi} |1> on first qubit, rest zeros
        amp0 = np.cos(thetas[i])
        amp1 = np.exp(1j * phis[i]) * np.sin(thetas[i])
        states[i, 0] = amp0
        states[i, 1] = amp1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset returning quantum states and targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.states[idx], dtype=torch.cfloat), torch.tensor(self.labels[idx], dtype=torch.float32)

class RegressionModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression model."""
    class QuantumLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, n_layers: int = 2):
            super().__init__()
            self.num_wires = num_wires
            self.n_layers = n_layers
            self.encoder = tq.AmplitudeEmbedding(embedding_dim=2**num_wires, wires=list(range(num_wires)))
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.cnot = tq.CNOT(has_params=False, trainable=False)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor):
            self.encoder(qdev, state_batch)
            for _ in range(self.n_layers):
                self.random_layer(qdev)
                for i in range(self.num_wires - 1):
                    self.cnot(qdev, wires=[i, i+1])
                self.rx(qdev)
                self.ry(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.quantum_layer = self.QuantumLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.quantum_layer.num_wires, bsz=bsz, device=state_batch.device)
        self.quantum_layer(qdev, state_batch)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["RegressionModel", "QuantumRegressionDataset", "generate_quantum_superposition_data"]
