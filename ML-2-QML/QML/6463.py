"""Quantum kernel construction using TorchQuantum with trainable entangling ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


class KernalAnsatz(tq.QuantumModule):
    """Entangling RY ansatz with trainable parameters.

    The circuit encodes two input vectors x and y in opposite directions
    using RY rotations with trainable offsets. A chain of CNOT gates
    entangles the qubits, allowing the kernel to capture correlations
    beyond simple product states.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Trainable offsets for each RY gate
        self.ry_offsets = nn.Parameter(torch.randn(n_wires))
        # CNOT entanglement pattern (chain)
        self.cnot_pairs = [(i, i + 1) for i in range(n_wires - 1)]

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Encode x and y into the quantum device.

        Parameters
        ----------
        q_device : tq.QuantumDevice
            Quantum device to apply gates on.
        x, y : torch.Tensor
            Input batches of shape (batch, n_wires).
        """
        batch_size = x.shape[0]
        q_device.reset_states(batch_size)

        # Encode x
        for i in range(self.n_wires):
            params = x[:, i] + self.ry_offsets[i]
            tq.ops.RY(q_device, wires=i, params=params)

        # Entangle
        for ctrl, tgt in self.cnot_pairs:
            tq.ops.CNOT(q_device, wires=[ctrl, tgt])

        # Encode y with negative sign
        for i in range(self.n_wires):
            params = -y[:, i] - self.ry_offsets[i]
            tq.ops.RY(q_device, wires=i, params=params)

        # Final entanglement
        for ctrl, tgt in reversed(self.cnot_pairs):
            tq.ops.CNOT(q_device, wires=[ctrl, tgt])


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of the encoded states."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap between encoded x and y.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape (batch, n_wires).

        Returns
        -------
        torch.Tensor
            Scalar overlap value for the first batch element.
        """
        self.ansatz(self.q_device, x, y)
        # Return the absolute amplitude of the |0...0⟩ component.
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors each of shape (n_wires,). They are first stacked
        into batches of shape (n, n_wires) and (m, n_wires) respectively.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = Kernel()
    a_stack = torch.stack([x.squeeze() for x in a])
    b_stack = torch.stack([x.squeeze() for x in b])
    return np.array([[kernel(x, y).item() for y in b_stack] for x in a_stack])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
