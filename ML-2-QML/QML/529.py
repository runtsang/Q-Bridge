"""Quantum kernel construction using PennyLane with a trainable variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as pnp
from typing import Sequence


class QuantumKernel(nn.Module):
    """
    Quantum RBF‑style kernel implemented with a variational circuit.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits in the device.
    layers : int, default=2
        Number of variational layers.
    dev_name : str, default="default.qubit"
        PennyLane device name.
    """

    def __init__(self, n_wires: int = 4, layers: int = 2, dev_name: str = "default.qubit") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.layers = layers
        self.dev = qml.device(dev_name, wires=n_wires)

        # Trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(layers, n_wires))

        # Build the QNode that returns the state vector
        @qml.qnode(self.dev, interface="torch")
        def _state_qnode(x: torch.Tensor) -> torch.Tensor:
            # Encode data
            for i in range(self.layers):
                for w in range(self.n_wires):
                    qml.RY(x[w], wires=w)
                # Entangling layer
                for w in range(self.n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                # Variational rotation
                for w in range(self.n_wires):
                    qml.RY(self.params[i, w], wires=w)
            return qml.state()

        self._state_qnode = _state_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape ``(n_wires,)``.  They are reshaped to ``(1, n_wires)`` for the
            simulator.

        Returns
        -------
        torch.Tensor
            Kernel value (scalar).
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        state_x = self._state_qnode(x.squeeze())
        state_y = self._state_qnode(y.squeeze())
        # Overlap magnitude squared
        return torch.abs(torch.dot(state_x.conj(), state_y)) ** 2

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Each element should be a 1‑D tensor of length ``n_wires``.

        Returns
        -------
        np.ndarray
            Kernel matrix as a NumPy array.
        """
        a_batch = torch.stack(a)
        b_batch = torch.stack(b)
        K = torch.zeros((len(a), len(b)), dtype=torch.float32)
        for i, x in enumerate(a_batch):
            for j, y in enumerate(b_batch):
                K[i, j] = self.forward(x, y)
        return K.cpu().numpy()


__all__ = ["QuantumKernel"]
