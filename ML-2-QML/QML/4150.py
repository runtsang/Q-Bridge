"""Quantum regression model that mirrors the classical architecture using TorchQuantum.

The model encodes input amplitudes, applies a variational random layer, optionally a fraud‑style circuit,
measures Pauli‑Z, and maps the expectation values to a scalar output.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SharedRegressionModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ):
        super().__init__()
        self.n_wires = num_wires
        # Generic Ry encoder per wire
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.fraud_layer = None
        if fraud_params is not None:
            # Build a simple fraud‑style variational block per wire
            fraud_modules = []
            for idx, params in enumerate(fraud_params):
                fraud_modules.append(
                    tq.BSgate(params.bs_theta, params.bs_phi, wires=tuple(range(num_wires)))
                )
                for wire, phase in enumerate(params.phases):
                    fraud_modules.append(tq.Rgate(phase, wires=wire))
                for wire, r in enumerate(params.squeeze_r):
                    fraud_modules.append(tq.Sgate(r, phase=0.0, wires=wire))
            self.fraud_layer = tq.QuantumModule()
            for i, mod in enumerate(fraud_modules):
                self.fraud_layer.add_module(f"fraud_{i}", mod)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        if self.fraud_layer is not None:
            self.fraud_layer(qdev)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "FraudLayerParameters",
    "generate_superposition_data",
    "RegressionDataset",
    "SharedRegressionModel",
]
