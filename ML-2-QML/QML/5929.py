"""Quantum kernel module with a variational ansatz.

The class mirrors the classical counterpart in API, enabling seamless
replacement in kernel‑based learning pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from typing import Sequence

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel module that implements a trainable variational circuit.
    The circuit consists of a configurable number of layers, each layer
    comprising Ry rotations followed by a ladder of CNOTs.  The overlap
    between the states produced by two inputs is used as the kernel
    value.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits used in the circuit.
    depth : int, default 3
        Number of variational layers.
    """

    def __init__(self, n_wires: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable parameters: shape (depth, n_wires)
        self.params = nn.Parameter(torch.randn(self.depth, self.n_wires))
        # Pre‑define the CNOT ladder pattern
        self.cnot_pattern = [(i, (i + 1) % self.n_wires) for i in range(self.n_wires)]

    def _apply_ansatz(self, x: torch.Tensor) -> None:
        """
        Apply the variational circuit to the device for input vector x.
        """
        self.q_device.reset_states(x.shape[0])
        # Encode data into Ry rotations
        for i in range(self.n_wires):
            tq.ry(self.q_device, wires=[i], params=x[:, i:i + 1])
        # Apply variational layers
        for d in range(self.depth):
            for i in range(self.n_wires):
                tq.ry(self.q_device, wires=[i], params=self.params[d, i : i + 1])
            for src, tgt in self.cnot_pattern:
                tq.cnot(self.q_device, wires=[src, tgt])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for two input batches.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape (N, n_wires).

        Returns
        -------
        torch.Tensor
            Kernel value of shape (N,).
        """
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        self._apply_ansatz(x)
        state_x = self.q_device.states.clone()
        self._apply_ansatz(y)
        state_y = self.q_device.states.clone()
        # Overlap magnitude squared
        overlap = torch.abs(torch.sum(state_x * torch.conj(state_y), dim=-1))
        return overlap

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : sequence of torch.Tensor
            Each element is a tensor of shape (n_wires,).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        return np.array(
            [
                [
                    self(
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32),
                    ).item()
                    for y in b
                ]
                for x in a
            ]
        )

    def circuit_depth(self) -> int:
        """
        Return the depth of the variational circuit in terms of two‑qubit gates.
        """
        # Each layer: n_wires Ry + n_wires CNOTs
        return self.depth * (self.n_wires + self.n_wires)

__all__ = ["QuantumKernelMethod"]
