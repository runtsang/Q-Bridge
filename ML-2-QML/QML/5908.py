"""Hybrid quantum regression model that combines a variational circuit with a quantum sampler.

The module defines a dataset, a quantum sampler, and a regression model that concatenates
features from a variational circuit and a sampler measurement.  The architecture is
fully differentiable via a quantum‑classical hybrid backend.

"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ}sinθ|1…1⟩."""
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

class HybridRegressionDataset(Dataset):
    """Dataset that returns quantum states and continuous labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNNQuantum(tq.QuantumModule):
    """Quantum sampler that applies a random layer and measures in the computational basis."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        return self.measure(qdev)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that concatenates variational circuit and sampler outputs."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=25, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.variational = self.QLayer(num_wires)
        self.sampler = SamplerQNNQuantum(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires * 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.variational(qdev)
        var_features = self.measure(qdev)
        sampler_features = self.sampler(qdev)
        features = torch.cat([var_features, sampler_features], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "HybridRegressionDataset", "generate_superposition_data", "SamplerQNNQuantum"]
