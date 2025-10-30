import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Sequence

# =========================
# Data generation
# =========================
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form cos(theta)|0…0> + exp(i phi) sin(theta)|1…1>
    and a non‑linear target.  The states are returned as complex numpy arrays.
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

# =========================
# Fraud‑detection style parameters
# =========================
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# =========================
# Quantum fraud layer
# =========================
class QuantumFraudLayer(tq.QuantumModule):
    """
    Quantum circuit that emulates a fraud‑style photonic layer using
    random gates and parameterised rotations.  The structure mirrors the
    classical fraud‑detection stack but operates on a quantum device.
    """
    def __init__(self, params: FraudLayerParameters, wires: Sequence[int]):
        super().__init__()
        self.params = params
        self.wires = wires
        # Random layer simulating the random feature map
        self.random = tq.RandomLayer(n_ops=10, wires=wires)
        # Parameterised rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        for w in self.wires:
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

# =========================
# Hybrid quantum regression model
# =========================
class HybridRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that mirrors the classical HybridRegressionModel.
    It encodes the input state, applies a series of fraud‑style quantum layers,
    a random quantum layer, measures Pauli‑Z, and feeds the result into a
    classical linear head.
    """
    def __init__(self, num_wires: int, fraud_params: Iterable[FraudLayerParameters]):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical features to amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Fraud‑style quantum layers
        self.fraud_layers = nn.ModuleList([
            QuantumFraudLayer(p, list(range(num_wires))) for p in fraud_params
        ])
        # Random quantum layer
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.fraud_layers:
            layer(qdev)
        self.random_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "HybridRegressionModel",
    "FraudLayerParameters",
    "QuantumFraudLayer",
    "generate_superposition_data",
]
