import pennylane as qml
import torch
from torch import nn
import numpy as np
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Variational quantum kernel using Pennylane. Encodes data via Ry gates and a trainable Rx layer."""
    def __init__(self, n_wires: int = 4, dev: str = "default.qubit"):
        super().__init__()
        self.n_wires = n_wires
        self.device = qml.device(dev, wires=n_wires)
        # trainable parameters for the variational layer
        self.ansatz_params = nn.Parameter(torch.randn(n_wires))
        # create a reusable QNode
        self._circuit = qml.QNode(self._qcircuit, self.device, interface="torch")

    def _qcircuit(self, data: torch.Tensor):
        # data encoding
        for i in range(self.n_wires):
            qml.RY(data[i], wires=i)
        # variational layer
        for i in range(self.n_wires):
            qml.RX(self.ansatz_params[i], wires=i)
        # return the full state vector
        return qml.state()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the fidelity between the encoded states of x and y."""
        state_x = self._circuit(x)
        state_y = self._circuit(y)
        # fidelity = |<x|y>|^2
        overlap = torch.abs(torch.vdot(state_x, state_y)) ** 2
        return overlap

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorised kernel matrix across two batches."""
        mat = torch.empty((a.shape[0], b.shape[0]))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Utility that returns a NumPy Gram matrix for a list of tensors."""
    qml_k = QuantumKernelMethod()
    mat = qml_k.kernel_matrix(torch.stack(a), torch.stack(b))
    return mat.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
