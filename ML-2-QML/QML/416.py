"""Quantum kernel implementation using Pennylane.

Author: The OpenAI‑GPT‑LLM‑Engine
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

import pennylane as qml

class QuantumKernelMethod(nn.Module):
    """
    Variational quantum kernel with a data‑encoding layer followed by a
    shallow circuit of parameterised rotations.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits in the device.  Defaults to 4.
    depth : int, optional
        Number of repetitions of the variational block.  Defaults to 2.
    dev_name : str, optional
        PennyLane device name.  Defaults to 'default.qubit'.
    wire_order : list[int], optional
        Order of wires used for the encoding.  If ``None`` a simple
        sequential order is used.

    Notes
    -----
    The kernel is defined as

        K(x, y) = |<ψ(x)|ψ(y)>|^2

    where |ψ(x)> is prepared by applying a sequence of RY rotations
    encoding the data followed by a shallow variational circuit.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2,
                 dev_name: str = "default.qubit",
                 wire_order: list[int] | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev_name = dev_name
        self.wire_order = wire_order or list(range(n_qubits))
        self.dev = qml.device(dev_name, wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def _state_qnode(x: torch.Tensor) -> torch.Tensor:
            """Prepare the state |ψ(x)>."""
            # Data‑encoding layer
            for i, wire in enumerate(self.wire_order):
                qml.RY(x[i], wires=wire)
            # Variational block
            for _ in range(self.depth):
                for wire in self.wire_order:
                    qml.RZ(0.1, wires=wire)
                for i in range(len(self.wire_order) - 1):
                    qml.CNOT(wires=[self.wire_order[i], self.wire_order[i + 1]])
            return qml.state()

        self._state_qnode = _state_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for two input vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of length ``n_qubits``.  They can be on CPU or CUDA.

        Returns
        -------
        torch.Tensor
            The scalar kernel value.
        """
        psi_x = self._state_qnode(x)
        psi_y = self._state_qnode(y)
        # Inner product magnitude squared
        overlap = torch.abs(torch.dot(psi_x.conj(), psi_y)) ** 2
        return overlap.squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_qubits: int = 4, depth: int = 2,
                  dev_name: str = "default.qubit") -> np.ndarray:
    """
    Compute the Gram matrix between two sets of vectors using the quantum kernel.

    Parameters
    ----------
    a, b : sequences of 1‑D torch tensors
        The datasets for which the kernel matrix is required.
    n_qubits : int, optional
        Number of qubits used in the quantum device.
    depth : int, optional
        Depth of the variational circuit.
    dev_name : str, optional
        PennyLane device name.

    Returns
    -------
    np.ndarray
        The kernel Gram matrix.
    """
    kernel = QuantumKernelMethod(n_qubits=n_qubits, depth=depth,
                                 dev_name=dev_name)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
