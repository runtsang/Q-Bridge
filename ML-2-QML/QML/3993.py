from __future__ import annotations

import torch
import numpy as np
import torch.quantum as tq
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

__all__ = ["FraudLayerParameters", "FraudRegressionHybrid", "RegressionDataset", "generate_superposition_data"]

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

def _apply_layer(qdev: tq.QuantumDevice, params: FraudLayerParameters, clip: bool) -> None:
    tq.BSgate(params.bs_theta, params.bs_phi, has_params=False, trainable=False)(qdev, wires=[0,1])
    for i, phase in enumerate(params.phases):
        tq.Rgate(phase, has_params=False, trainable=False)(qdev, wires=[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        tq.Sgate(r if not clip else _clip(r, 5), phi, has_params=False, trainable=False)(qdev, wires=[i])
    tq.BSgate(params.bs_theta, params.bs_phi, has_params=False, trainable=False)(qdev, wires=[0,1])
    for i, phase in enumerate(params.phases):
        tq.Rgate(phase, has_params=False, trainable=False)(qdev, wires=[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        tq.Dgate(r if not clip else _clip(r, 5), phi, has_params=False, trainable=False)(qdev, wires=[i])
    for i, k in enumerate(params.kerr):
        tq.Kgate(k if not clip else _clip(k, 1), has_params=False, trainable=False)(qdev, wires=[i])

class FraudRegressionHybrid(tq.QuantumModule):
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

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters], num_wires: int):
        super().__init__()
        self.fraud_params = [input_params] + list(layers)
        self.num_wires = num_wires
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=2, bsz=bsz, device=state_batch.device)
        encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2xRy"])
        encoder(qdev, state_batch)
        for params in self.fraud_params:
            _apply_layer(qdev, params, clip=True)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

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
