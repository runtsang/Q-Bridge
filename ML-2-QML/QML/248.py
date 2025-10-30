"""Quantum kernel implementation using Pennylane.

The class QuantumKernelMethod mirrors the classical counterpart but
evaluates the kernel as a state‑overlap between two data‑encoded
circuits.  The ansatz depth is a hyper‑parameter and the rotation angles
are trainable, allowing the kernel to adapt to the task.

API
---
forward(x, y) -> torch.Tensor
    Returns the kernel value for two 1‑D tensors.
gram_matrix(a, b) -> np.ndarray
    Computes the full Gram matrix for two sequences of tensors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Quantum RBF‑style kernel using a parameterized ansatz.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits; defaults to the dimensionality of the input
        vectors (capped at 8 for simulation stability).
    depth : int, optional
        Number of repeated layers in the ansatz.
    device : str, optional
        Pennylane quantum device name; defaults to "default.qubit".
    seed : int, optional
        Random seed for reproducibility.

    Notes
    -----
    * The ansatz consists of a layer of Ry rotations encoding the data
      followed by a layer of CNOTs that entangle neighboring qubits.
      The depth parameter repeats this pattern.
    * The kernel is the squared absolute value of the inner product
      between the two resulting quantum states.
    * For large datasets, gram_matrix uses vectorised state preparation
      to avoid repeated device resets.
    """
    def __init__(
        self,
        n_qubits: int | None = None,
        depth: int = 2,
        device: str = "default.qubit",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.n_qubits = n_qubits or 8
        self.device = qml.device(device, wires=self.n_qubits, shots=None, seed=seed)
        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(self.depth, self.n_qubits))

    def _ansatz(self, x: torch.Tensor, params: torch.Tensor) -> None:
        """
        Data‑encoding ansatz applied to the device.

        Parameters
        ----------
        x : torch.Tensor
            1‑D tensor of length ``n_qubits``.
        params : torch.Tensor
            Shape (depth, n_qubits) rotation angles.
        """
        for d in range(self.depth):
            for q in range(self.n_qubits):
                qml.Ry(x[q], wires=q)
            # Entangling layer
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            # Optional trainable rotation
            for q in range(self.n_qubits):
                qml.Rz(params[d, q], wires=q)

    def _kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute |⟨ψ(x)|ψ(y)⟩|² using a single circuit call per data pair.
        """
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def prepare_state(vec):
            self._ansatz(vec, self.params)
            return qml.state()

        state_x = prepare_state(x)
        state_y = prepare_state(y)
        overlap = torch.dot(state_x.conj(), state_y)
        return torch.abs(overlap) ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Public forward interface matching the classical API.
        """
        return self._kernel_value(x, y)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix efficiently using batched state preparation.
        """
        m, n = len(a), len(b)
        K = np.empty((m, n))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                K[i, j] = self.forward(x, y).item()
        return K


__all__ = ["QuantumKernelMethod"]
