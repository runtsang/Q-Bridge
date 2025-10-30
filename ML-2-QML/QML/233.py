import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATExtendedQML(nn.Module):
    """
    Hybrid classical‑quantum model: a CNN encoder feeding a variational quantum circuit.
    The circuit outputs 4 expectation values (Pauli‑Z) that are batch‑normalized.
    """
    def __init__(self, n_wires: int = 4, device: str = "cpu"):
        super().__init__()
        self.n_wires = n_wires
        # Classical encoder identical to the first part of the ML model
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_wires, shots=None)
        # Learnable variational parameters
        self.theta = nn.Parameter(torch.randn(n_wires, 3))
        self.rho   = nn.Parameter(torch.randn(n_wires, 3))
        self.norm  = nn.BatchNorm1d(n_wires)

    def _quantum_circuit(self, features: torch.Tensor):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit():
            # Feature encoding via Y‑rotations
            for i in range(self.n_wires):
                qml.RY(features[i], wires=i)
            # First variational block
            for i in range(self.n_wires):
                qml.RY(self.theta[i, 0], wires=i)
                qml.RZ(self.theta[i, 1], wires=i)
                qml.RX(self.theta[i, 2], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second variational block
            for i in range(self.n_wires):
                qml.RY(self.rho[i, 0], wires=i)
                qml.RZ(self.rho[i, 1], wires=i)
                qml.RX(self.rho[i, 2], wires=i)
            # Final entangling
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i + 1, i])
            # Readout: expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]
        return circuit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.encoder(x)          # (bsz, 16, 1, 1)
        feat = self.flatten(feat)       # (bsz, 16)
        feat = feat[:, :self.n_wires]   # (bsz, n_wires)
        # Scale to [-π, π] for rotation angles
        scaled = (feat - feat.mean()) / (feat.std() + 1e-6) * torch.pi
        # Apply quantum circuit to each batch element
        out = torch.stack([self._quantum_circuit(scaled[i]) for i in range(bsz)], dim=0)
        return self.norm(out)

__all__ = ["QuantumNATExtendedQML"]
