"""Quantum kernel with a variational ansatz that can be trained.

The implementation builds on TorchQuantum and extends the original
`Kernel` by:
- a learnable set of rotation angles (one per wire per layer);
- support for arbitrary depth and gate type;
- a `kernel_matrix` method compatible with the classical API.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn


class QuantumKernelMethod(tq.QuantumModule):
    """
    Variational quantum kernel.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits used for the circuit.
    depth : int, default=2
        Number of layers in the variational ansatz.
    gate : str, default="ry"
        Single‑qubit rotation gate used in the ansatz.
    trainable : bool, default=True
        Whether the circuit parameters should be optimized.
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 2,
        gate: str = "ry",
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.gate = gate
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Parameter matrix: shape (depth, n_wires)
        if trainable:
            self.params = nn.Parameter(torch.randn(depth, n_wires))
        else:
            self.register_buffer("params", torch.randn(depth, n_wires))

    def _gate_func(self, wire: int, angle: torch.Tensor) -> None:
        """Apply a single‑qubit rotation gate to a specified wire."""
        if self.gate == "ry":
            tq.ops.ry(self.q_device, wires=[wire], params=angle)
        elif self.gate == "rz":
            tq.ops.rz(self.q_device, wires=[wire], params=angle)
        else:
            raise ValueError(f"Unsupported gate: {self.gate}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum kernel between two batches of samples.

        Parameters
        ----------
        x, y : torch.Tensor
            Batches of shape (batch, features). The number of features must
            match ``n_wires``.
        Returns
        -------
        torch.Tensor
            Kernel value of shape (batch,).
        """
        batch_size = x.shape[0]
        self.q_device.reset_states(batch_size)

        # Encode x into the circuit
        for l in range(self.depth):
            for w in range(self.n_wires):
                angle = self.params[l, w] * x[:, w]
                self._gate_func(w, angle)

        # Encode y in reverse order with negative sign
        for l in reversed(range(self.depth)):
            for w in range(self.n_wires):
                angle = -self.params[l, w] * y[:, w]
                self._gate_func(w, angle)

        # Return the absolute value of the first amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix for two collections of samples.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of tensors that will be stacked into 2‑D arrays.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        K = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                K[i, j] = self.forward(x, y).item()
        return K

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_wires={self.n_wires}, depth={self.depth}, gate='{self.gate}')"


__all__ = ["QuantumKernelMethod"]
