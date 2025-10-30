import torch
import torch.nn as nn
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

class QFCModel(nn.Module):
    """Quantum variational circuit for 4‑class classification."""
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_wires = 4
        self.n_layers = n_layers
        # Parameters for rotation gates in each layer
        self.params = nn.Parameter(0.01 * torch.randn(n_layers, self.n_wires, 3))

    def circuit(self, x: torch.Tensor) -> torch.Tensor:
        # Encode classical input with Ry gates
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
        # Variational layers
        for l in range(self.n_layers):
            for i in range(self.n_wires):
                qml.Rot(*self.params[l, i], wires=i)
            # Entangling pattern
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
        # Measurement in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    @qml.qnode(dev, interface="torch")
    def qnode(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)

__all__ = ["QFCModel"]
