from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Tuple

# Dataset generation (quantum superposition)
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
    return states, labels

# Fraud‑style parameters for photonic gates
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# Quantum layer emulating a photonic fraud layer
class FraudQLayer(tq.QuantumModule):
    def __init__(self, num_wires: int, params: FraudLayerParameters):
        super().__init__()
        self.n_wires = num_wires
        self.params = params
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.bs = tq.BSgate
        self.rgate = tq.RZ
        self.sgate = tq.Sgate
        self.dgate = tq.Dgate
        self.kgate = tq.Kgate

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        # Beam‑splitter on first two wires
        self.bs(self.params.bs_theta, self.params.bs_phi) | (qdev.wires[0], qdev.wires[1])
        for i, phase in enumerate(self.params.phases):
            self.rgate(phase) | qdev.wires[i]
        for i, (r, phi) in enumerate(zip(self.params.squeeze_r, self.params.squeeze_phi)):
            self.sgate(_clip(r, 5), phi) | qdev.wires[i]
        for i, (r, phi) in enumerate(zip(self.params.displacement_r, self.params.displacement_phi)):
            self.dgate(_clip(r, 5), phi) | qdev.wires[i]
        for i, k in enumerate(self.params.kerr):
            self.kgate(_clip(k, 1)) | qdev.wires[i]

class RegressionModel(tq.QuantumModule):
    def __init__(self, num_wires: int, fraud_params: FraudLayerParameters):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = FraudQLayer(num_wires, fraud_params)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "FraudLayerParameters",
    "FraudQLayer",
    "RegressionDataset",
    "RegressionModel",
    "generate_superposition_data",
]
