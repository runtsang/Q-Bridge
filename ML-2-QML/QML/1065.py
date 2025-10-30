"""Variational quantum kernel approximating an RBF kernel using Pennylane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """
    Quantum kernel implemented with a variational ansatz and amplitude encoding.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits for the amplitude‑encoded state.
    n_layers : int, default=3
        Depth of the variational circuit.
    device : str, default='default.qubit'
        Pennylane device name.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        # Trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        # QNode that returns the overlap between two amplitude‑encoded states
        @qml.qnode(self.dev, interface="torch")
        def _kernel_qnode(x: torch.Tensor, y: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Amplitude encode the first vector
            qml.AmplitudeEmbedding(
                features=x,
                wires=range(n_qubits),
                normalize=True,
            )
            # Apply the variational ansatz
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.Rot(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], wires=qubit)
            # Uncompute the first state
            for qubit in range(n_qubits):
                qml.Adjoint(qml.AmplitudeEmbedding)(features=x, wires=range(n_qubits), normalize=True)
            # Amplitude encode the second vector
            qml.AmplitudeEmbedding(
                features=y,
                wires=range(n_qubits),
                normalize=True,
            )
            # Apply the same variational ansatz
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.Rot(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], wires=qubit)
            # Uncompute the second state
            for qubit in range(n_qubits):
                qml.Adjoint(qml.AmplitudeEmbedding)(features=y, wires=range(n_qubits), normalize=True)
            # Measure the overlap
            return qml.expval(qml.PauliZ(0))

        self._kernel_qnode = _kernel_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape ``(d,)`` or ``(n, d)``.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        # Ensure input dimensionality matches qubits
        assert x.shape[1] == self.n_qubits, "Input dimension must equal number of qubits."
        assert y.shape[1] == self.n_qubits, "Input dimension must equal number of qubits."
        return self._kernel_qnode(x, y, self.params)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two datasets using the quantum kernel.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of input vectors.
        """
        a_tensor = torch.stack(a).float()
        b_tensor = torch.stack(b).float()
        gram = torch.zeros((len(a), len(b)))
        for i, x in enumerate(a_tensor):
            for j, y in enumerate(b_tensor):
                gram[i, j] = self.forward(x, y)
        return gram.detach().numpy()


__all__ = ["QuantumKernel"]
