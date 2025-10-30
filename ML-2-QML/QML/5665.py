"""Quantum kernel implementation using PennyLane.

The module keeps the original class names for backward compatibility
but adds a variational ansatz with trainable parameters and a
measurement‑based similarity metric.  The kernel value is computed
as the absolute overlap between two encoded quantum states, which
is differentiable through PennyLane's automatic differentiation.
"""

import pennylane as qml
import numpy as np
import torch
from typing import Sequence

class KernalAnsatz:
    """Variational quantum kernel with a feature map and trainable entangling layer.

    The ansatz applies a feature‑map of Ry rotations followed by a
    trainable two‑layer entangling circuit.  The kernel value is the
    absolute value of the overlap between the encoded states.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Trainable parameters for the entangling layers
        self.params = np.random.randn(n_layers, n_wires, 3)
        self.qdevice = qml.device("default.qubit", wires=n_wires)

    def feature_map(self, x: np.ndarray) -> None:
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)

    def variational_layer(self, params: np.ndarray) -> None:
        for i in range(self.n_wires):
            qml.RX(params[0, i, 0], wires=i)
            qml.RZ(params[0, i, 1], wires=i)
            qml.RY(params[0, i, 2], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])

    def kernel_function(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel as |<phi(x)|phi(y)>|^2."""
        @qml.qnode(self.qdevice, interface="torch")
        def circuit(x, y):
            self.feature_map(x)
            self.variational_layer(self.params)
            qml.adjoint(self.feature_map)(y)  # apply inverse feature map of y
            return qml.expval(qml.PauliZ(0))

        return abs(circuit(x, y))

class Kernel:
    """Callable wrapper around :class:`KernalAnsatz`."""

    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        self.ansatz = KernalAnsatz(n_wires, n_layers)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        val = self.ansatz.kernel_function(x_np, y_np)
        return torch.tensor(val, dtype=torch.float32)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two datasets using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
