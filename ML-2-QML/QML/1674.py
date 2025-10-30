"""Quantum kernel based on a variational Ry‑rotation ansatz.

The kernel returns the probability of measuring |0…0> after
applying the encoding of x followed by the inverse encoding of y.
This probability equals the squared magnitude of the overlap
between the two encoded states.
"""

from __future__ import annotations

import pennylane as qml
import torch
import numpy as np
from typing import Iterable

class KernelMethod:
    """
    Quantum kernel using a parameterised Ry rotation ansatz.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits.
    device_name : str, default 'default.qubit'
        PennyLane device name.
    """

    def __init__(self,
                 n_wires: int = 4,
                 device_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.dev = qml.device(device_name, wires=self.n_wires)

    def _kernel_qnode(self, x_dat: np.ndarray, y_dat: np.ndarray):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, y):
            # Encode first vector
            for i in range(self.n_wires):
                qml.Ry(x[i], wires=i)
            # Encode second vector with negative sign
            for i in range(self.n_wires):
                qml.Ry(-y[i], wires=i)
            # Return probability of |0...0>
            return qml.probs(wires=range(self.n_wires))[0]
        return circuit(x_dat, y_dat)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for two single data points.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (n_wires,).
        """
        x_np = np.array(x.tolist(), dtype=np.float32)
        y_np = np.array(y.tolist(), dtype=np.float32)
        val = self._kernel_qnode(x_np, y_np)
        return torch.abs(val)

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the Gram matrix between two collections of tensors.

        Parameters
        ----------
        a, b : iterable of torch.Tensor
            Each yields a tensor of shape (n_wires,).
        """
        a_list = list(a)
        b_list = list(b)
        mat = torch.zeros((len(a_list), len(b_list)), dtype=torch.float32)
        for i, x in enumerate(a_list):
            for j, y in enumerate(b_list):
                mat[i, j] = self.forward(x, y)
        return mat

__all__ = ["KernelMethod"]
