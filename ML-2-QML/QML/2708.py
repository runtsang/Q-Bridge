"""Hybrid quantum regression model.

This module extends the original QuantumRegression.py by adding a
parameterised quantum layer inspired by the fraud‑detection photonic
circuit.  The model remains fully quantum‑classical: a quantum encoder,
a stack of RandomLayer + RX+RY gates, followed by a measurement and a
classical linear head.

The dataset generator and RegressionDataset are identical to the
original example for compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Data generation (identical to original)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a quantum superposition dataset.

    The function is unchanged from the original example to preserve API
    compatibility.
    """
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

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class RegressionDataset(torch.utils.data.Dataset):
    """Quantum dataset compatible with the original API."""
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
# Fraud‑style quantum layer definition
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑style quantum block."""
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

def _apply_layer(qdev: tq.QuantumDevice, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a fraud‑style photonic circuit to a quantum device."""
    # Beam splitter
    tq.BSgate(params.bs_theta, params.bs_phi)(qdev, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        tq.Rgate(phase)(qdev, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        tq.Sgate(r if not clip else _clip(r, 5), phi)(qdev, wires=i)
    # Second beam splitter
    tq.BSgate(params.bs_theta, params.bs_phi)(qdev, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        tq.Rgate(phase)(qdev, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        tq.Dgate(r if not clip else _clip(r, 5), phi)(qdev, wires=i)
    for i, k in enumerate(params.kerr):
        tq.Kgate(k if not clip else _clip(k, 1))(qdev, wires=i)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> tq.QuantumModule:
    """Return a quantum module that applies a stack of fraud‑style layers."""
    class FraudModule(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.input_params = input_params
            self.layer_params = list(layers)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            _apply_layer(qdev, self.input_params, clip=False)
            for params in self.layer_params:
                _apply_layer(qdev, params, clip=True)

    return FraudModule()

# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #

class QModel(tq.QuantumModule):
    """Quantum regression model that mirrors the classical architecture."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        # Quantum encoder: simple random layer to embed the input state
        self.encoder = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        # Fraud‑style quantum layers
        def _rand_param() -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=np.random.randn(),
                bs_phi=np.random.randn(),
                phases=tuple(np.random.randn(2)),
                squeeze_r=tuple(np.random.randn(2)),
                squeeze_phi=tuple(np.random.randn(2)),
                displacement_r=tuple(np.random.randn(2)),
                displacement_phi=tuple(np.random.randn(2)),
                kerr=tuple(np.random.randn(2)),
            )
        input_params = _rand_param()
        layer_params = [_rand_param() for _ in range(num_layers - 1)]
        self.fraud_net = build_fraud_detection_program(input_params, layer_params)
        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state
        self.encoder(qdev, state_batch)
        # Apply fraud‑style layers
        self.fraud_net(qdev)
        # Measure
        features = self.measure(qdev)
        # Classical head
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "FraudLayerParameters"]
