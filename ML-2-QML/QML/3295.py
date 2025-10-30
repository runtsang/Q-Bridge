"""Quantum hybrid regression model that incorporates fraud‑detection style parameters as gate angles.

The circuit starts with a general linear encoder, followed by a randomised layer and a
sequence of RX/RY rotations whose angles are derived from the FraudLayerParameters.
The output of the measurement is mapped to a scalar via a classical linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, List, Sequence


# --------------------------------------------------------------------------- #
#  Dataset utilities – identical to the original QuantumRegression seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Simple PyTorch dataset yielding quantum state vectors and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Fraud‑detection inspired parameters for quantum gates
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that will be mapped to rotation angles in the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
#  Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that uses fraud‑detection parameters as rotation angles."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, fraud_params: List[FraudLayerParameters]):
            super().__init__()
            self.n_wires = num_wires
            self.fraud_params = fraud_params
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for params in self.fraud_params:
                # Map fraud parameters to rotation angles; clip for numerical stability
                rx_angle = _clip(params.bs_theta, 5.0)
                ry_angle = _clip(params.bs_phi, 5.0)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire, params=rx_angle)
                    self.ry(qdev, wires=wire, params=ry_angle)

    def __init__(self, num_wires: int, fraud_params: List[FraudLayerParameters]):
        super().__init__()
        self.n_wires = num_wires
        # A generic linear encoder that maps the input state to a product state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, fraud_params)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "FraudLayerParameters"]
