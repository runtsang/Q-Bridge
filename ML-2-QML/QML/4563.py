import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for API compatibility)."""
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

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Returns complex state vectors and target labels.
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

class FraudDetectionQuantumModel(nn.Module):
    """
    Variational quantum circuit inspired by the photonic fraud detection example.
    Encodes 2‑dimensional inputs with RY rotations, applies a depth‑controlled
    variational layer of RX/RY and CNOT entanglement, measures Pauli‑Z, and
    feeds the expectation values to a classical regression head.
    """
    def __init__(self, num_wires: int, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.qnode = qml.QNode(self._qcircuit, self.dev, interface="torch")
        self.head = nn.Linear(num_wires, 1)

    def _qcircuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Feature encoding with RY rotations
        for i in range(self.num_wires):
            qml.RY(x[i], wires=i)
        # Variational ansatz
        for l in range(self.num_layers):
            for i in range(self.num_wires):
                qml.RX(params[l, i], wires=i)
                qml.RY(params[l, i], wires=i)
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(wires=list(range(self.num_wires))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Randomly initialise variational parameters per sample
        params = torch.randn(self.num_layers, self.num_wires, requires_grad=True, device=x.device)
        out = torch.stack([self.qnode(x[i], params) for i in range(batch_size)])
        return self.head(out)

__all__ = [
    "FraudDetectionQuantumModel",
    "generate_superposition_data",
]
