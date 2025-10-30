"""Quantum model for Quantum‑NAT with a variational circuit and measurement."""
import pennylane as qml
import torch
import torch.nn as nn

# Quantum device with 4 wires
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def circuit(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Encoding: rotate each wire by the corresponding input feature
    for i in range(4):
        qml.RY(x[i], wires=i)
    # Entangling layer
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])
    # Parameterized rotations and entanglement
    for layer in range(2):
        for j in range(4):
            qml.RZ(params[layer, j], wires=j)
        for j in range(3):
            qml.CNOT(wires=[j, j + 1])
    # Return expectation values of Pauli‑Z on each wire
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuantumNATEnhanced(nn.Module):
    """Quantum model with a variational circuit and measurement."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Two layers of rotation parameters
        self.params = nn.Parameter(torch.randn(2, self.n_wires))
        self.norm = nn.BatchNorm1d(self.n_wires)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, 4) after classical pooling.
        """
        batch_size = x.shape[0]
        # Stack outputs from the quantum circuit for each batch element
        out = torch.stack([circuit(self.params, x[i]) for i in range(batch_size)])
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
