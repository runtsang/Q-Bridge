"""Quantum regression model with entangling variational layers and dual‑measurement head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(
    num_wires: int,
    samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and labels for quantum regression."""
    rng = np.random.default_rng(seed)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = rng.random(samples) * 2 * np.pi
    phis = rng.random(samples) * 2 * np.pi
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for quantum states."""

    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression(tq.QuantumModule):
    """Variational quantum circuit for regression with entanglement and multi‑measurement."""

    class EntanglingLayer(tq.QuantumModule):
        """Entangles wires with a repeated RX‑RY block followed by a CNOT."""

        def __init__(self, n_wires: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = n_wires
            self.layer_list = nn.ModuleList()
            for _ in range(n_layers):
                layer = nn.ModuleList()
                for wire in range(n_wires):
                    layer.append(tq.RX(has_params=True, trainable=True))
                    layer.append(tq.RY(has_params=True, trainable=True))
                layer.append(tq.CNOT(wires=(0, 1)))  # simple entangler
                self.layer_list.append(layer)

        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.layer_list:
                for gate in layer:
                    if isinstance(gate, tq.CNOT):
                        gate(qdev, wires=gate.wires)
                    else:
                        gate(qdev)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.entangler = self.EntanglingLayer(num_wires)
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.entangler(qdev)
        feat_z = self.measure_z(qdev)
        feat_x = self.measure_x(qdev)
        features = torch.cat([feat_z, feat_x], dim=-1)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
