"""Quantum kernel implementation using Pennylane.

This version defines a variational feature map that can be tuned
via the depth parameter.  The kernel evaluates the squared state
overlap between two feature vectors on a default qubit simulator.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from typing import Sequence

class QuantumKernelMethod:
    """Quantum kernel with variational feature map.

    Parameters
    ----------
    depth : int, default 2
        Number of layers in the variational ansatz.
    wires : int | None, default None
        Number of qubits.  If None, inferred from the feature dimension.
    """
    def __init__(self, depth: int = 2, wires: int | None = None) -> None:
        self.depth = depth
        self.wires = wires
        self.dev = None
        self.qnode = None

    def _build_ansatz(self, x: np.ndarray) -> None:
        """Define the variational circuit."""
        dev = qml.device("default.qubit", wires=self.wires)

        @qml.qnode(dev)
        def circuit(params):
            for i in range(self.wires):
                qml.RY(x[i], wires=i)
            for d in range(self.depth):
                for i in range(self.wires):
                    qml.RZ(params[d, i], wires=i)
                for i in range(self.wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self.qnode = circuit
        self.dev = dev

    def fit(self, X: torch.Tensor) -> None:
        """Prepare the device and ansatz based on input dimension."""
        n_features = X.shape[1]
        self.wires = n_features
        self._build_ansatz(X[0].numpy())

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor] | None = None) -> np.ndarray:
        """Compute the Gram matrix via state overlap."""
        if Y is None:
            Y = X
        n = len(X)
        m = len(Y)
        K = np.zeros((n, m))
        for i, xi in enumerate(X):
            psi_x = self.qnode(xi.numpy())
            for j, yj in enumerate(Y):
                psi_y = self.qnode(yj.numpy())
                K[i, j] = np.abs(np.vdot(psi_x, psi_y)) ** 2
        return K

__all__ = ["QuantumKernelMethod"]
