"""Quantum regression model with richer entanglement and hybrid classical head.

- Adds a parameterized entanglement layer and multiple measurement operators.
- Supports noise injection in data generation for robustness studies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states with optional Gaussian noise."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, size=states.shape) + 1j * np.random.normal(0, noise_level, size=states.shape)
        states += noise

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Hybrid quantum‑classical regression network."""

    class QLayer(tq.QuantumModule):
        """Parameterised circuit with entanglement and single‑qubit rotations."""

        def __init__(self, num_wires: int, n_layers: int = 3):
            super().__init__()
            self.n_wires = num_wires
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                # Random rotations
                self.layers.append(tq.RandomLayer(n_ops=20, wires=list(range(num_wires))))
                # Entanglement via CNOT chain
                for i in range(num_wires):
                    self.layers.append(tq.CNOT(wires=(i, (i + 1) % num_wires)))
                # Trainable single‑qubit rotations
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RY(has_params=True, trainable=True))

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in self.layers:
                layer(qdev)

    def __init__(
        self,
        num_wires: int,
        classical_hidden: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        # Measure both Z and X to enrich feature set
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        # Classical head to map quantum features to a scalar
        if classical_hidden is None:
            classical_hidden = [32, 16]
        head_layers = []
        input_dim = num_wires * 2  # Z and X measurements
        for hidden in classical_hidden:
            head_layers.append(nn.Linear(input_dim, hidden))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout))
            input_dim = hidden
        head_layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        z_features = self.measure_z(qdev)
        x_features = self.measure_x(qdev)
        features = torch.cat([z_features, x_features], dim=1)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
