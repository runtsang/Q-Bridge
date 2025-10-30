"""Quantum kernel using Pennylane."""
import pennylane as qml
import torch
from typing import Sequence
import numpy as np

class HybridKernel(torch.nn.Module):
    """Quantum kernel with a parameterised feature map."""
    def __init__(self, num_wires: int = 4, entanglement: str = "cnot"):
        super().__init__()
        self.num_wires = num_wires
        self.entanglement = entanglement
        self.dev = qml.device("default.qubit", wires=self.num_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            for i in range(self.num_wires):
                qml.RY(x[i], wires=i)
            if self.entanglement == "cnot":
                for i in range(self.num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self.circuit = circuit

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel between two samples."""
        x = x.flatten()
        y = y.flatten()
        state_x = self.circuit(x)
        state_y = self.circuit(y)
        fidelity = torch.abs(torch.dot(state_x, torch.conj(state_y))) ** 2
        return fidelity

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  num_wires: int = 4) -> np.ndarray:
    """Compute Gram matrix between two lists of tensors using the quantum kernel."""
    kernel = HybridKernel(num_wires)
    mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            mat[i, j] = kernel(xi, yj)
    return mat.detach().cpu().numpy()

__all__ = ["HybridKernel", "kernel_matrix"]
