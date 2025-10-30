import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import List

# Photonic‑style parameters reused from the FraudDetection example
@dataclass
class FraudParams:
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

def _photonic_layer(q, params: FraudParams, clip: bool = False):
    """Apply a photonic‑style layer using Pennylane ops."""
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=i)
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=i)
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=i)

class HybridNATModel(nn.Module):
    """Hybrid classical‑quantum model with photonic‑style quantum layer."""
    def __init__(self,
                 n_qubits: int = 4,
                 n_photonic_layers: int = 2,
                 fraud_params: List[FraudParams] | None = None):
        super().__init__()
        # Classical CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_dim = 16 * 7 * 7

        # Quantum device and circuit
        self.device = qml.device("default.qubit", wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_photonic_layers = n_photonic_layers
        self.fraud_params = fraud_params or [FraudParams(0.5,0.5,(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0),(0.0,0.0)) for _ in range(n_photonic_layers)]

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def quantum_circuit(x):
            # Encode classical features into rotations
            for i in range(n_qubits):
                qml.RY(x[:, i] if x.shape[1] > i else 0, wires=i)
            # Stack photonic‑style layers
            for idx, params in enumerate(self.fraud_params):
                _photonic_layer(qml, params, clip=(idx > 0))
            return qml.expval(qml.PauliZ(0))

        self.quantum_circuit = quantum_circuit

        # Fully‑connected head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)
        flat = features.view(bsz, -1)
        # Prepare input for quantum circuit
        # Pad with zeros if necessary
        q_input = torch.zeros(bsz, self.n_qubits, device=x.device)
        q_input[:, :flat.shape[1]] = flat[:, :self.n_qubits]
        q_out = self.quantum_circuit(q_input)
        q_out = q_out.unsqueeze(-1)
        combined = torch.cat([flat, q_out], dim=1)
        out = self.head(combined)
        return self.norm(out)

__all__ = ["HybridNATModel", "FraudParams"]
