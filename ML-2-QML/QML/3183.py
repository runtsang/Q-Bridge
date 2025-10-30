from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudDetectionHybrid(tq.QuantumModule):
    """
    Quantum‑classical hybrid fraud‑detection model.
    The encoder maps classical inputs into a superposition state, a random
    variational layer processes the state, and the output is read out with
    Pauli‑Z measurements and fed to a classical head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 2, num_features: int = 2):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_features}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

def build_fraud_detection_program_quantum(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    num_wires: int = 2,
) -> tq.QuantumDevice:
    """Build a quantum program that emulates the photonic‑layer structure."""
    qdev = tq.QuantumDevice(n_wires=num_wires, bsz=1, device="cpu")
    for layer in layers:
        # Emulate photonic clipping with a random layer
        tq.RandomLayer(n_ops=5, wires=list(range(num_wires)))(qdev)
    return qdev

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and binary fraud labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = (np.sin(2 * thetas) * np.cos(phis) > 0).astype(np.float32)
    return states, labels

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_program_quantum", "generate_superposition_data"]
