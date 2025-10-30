from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn
import numpy as np

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

class QuantumFraudDetectionHybrid(nn.Module):
    """Quantum‑enhanced fraud‑detection model.  Classical photonic layers are
    implemented with PyTorch; the quantum part is a Pennylane QNode
    that maps the classical representation into a small quantum register
    and measures a Pauli‑Z expectation that is fed into a classical head."""
    def __init__(self, fraud_params: Iterable[FraudLayerParameters], n_qubits: int = 4):
        super().__init__()
        modules = []
        for i, param in enumerate(fraud_params):
            modules.append(_layer_from_params(param, clip=(i > 0)))
        self.classical = nn.Sequential(*modules)
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.var_params = nn.Parameter(torch.randn(2 * n_qubits))
        self.head = nn.Linear(n_qubits, 1)

        def _circuit(z, theta):
            for i in range(self.n_qubits):
                qml.RY(z[i], wires=i)
            for i in range(self.n_qubits):
                qml.RX(theta[2 * i], wires=i)
                qml.RZ(theta[2 * i + 1], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.quantum_circuit = qml.qnode(self.dev, interface="torch")(_circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classic = self.classical(x)
        batch = classic.shape[0]
        features = classic.view(batch, -1)
        q_outputs = []
        for i in range(batch):
            q_outputs.append(self.quantum_circuit(features[i], self.var_params))
        q_tensor = torch.stack(q_outputs)
        return self.head(q_tensor)

__all__ = ["FraudLayerParameters", "QuantumFraudDetectionHybrid", "generate_superposition_data"]
