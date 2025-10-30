"""Quantum regression model with entanglement and multi‑observable readout.

Includes configurable amplitude encoding, a randomized entanglement layer,
parameterised rotations, and measurement of Z, X, Y expectation values.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^(i phi) sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Torch Dataset for quantum states and target values."""

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
    """Hybrid quantum‑classical regression model."""

    class QLayer(tq.QuantumModule):
        """Entangled variational layer."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.entangle = tq.RandomLayer(
                n_ops=10,
                wires=list(range(num_wires)),
                has_params=False,
            )
            self.param_layer = tq.RandomLayer(
                n_ops=3 * num_wires,
                wires=list(range(num_wires)),
                has_params=True,
            )

        def forward(self, qdev: tq.QuantumDevice):
            self.entangle(qdev)
            self.param_layer(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Amplitude encoding using Ry rotations
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        # Measure expectation values of Z, X, Y on each wire
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.measure_y = tq.MeasureAll(tq.PauliY)
        self.head = nn.Linear(3 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        z = self.measure_z(qdev)
        x = self.measure_x(qdev)
        y = self.measure_y(qdev)
        features = torch.cat([z, x, y], dim=-1)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
