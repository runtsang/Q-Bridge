"""Quantum kernel construction with a learnable variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from typing import Sequence

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(tq.QuantumModule):
    """Parameterized ansatz that encodes two classical vectors.

    The circuit depth and rotation angles are learnable parameters.
    """

    def __init__(self, n_wires: int, depth: int):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        init_params = torch.randn(depth, n_wires)
        self.params = nn.Parameter(init_params)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode ``x`` with positive rotations and ``y`` with negative rotations
        in reverse order, interleaved with a fixed entangling layer.
        """
        q_device.reset_states(x.shape[0])

        # Positive encoding of x
        for d in range(self.depth):
            for w in range(self.n_wires):
                param = x[:, w] * self.params[d, w]
                tq.ry(q_device, wires=[w], params=param)

        # Negative encoding of y (reverse order)
        for d in reversed(range(self.depth)):
            for w in range(self.n_wires):
                param = -y[:, w] * self.params[d, w]
                tq.ry(q_device, wires=[w], params=param)

        # Entangling layer
        for w in range(self.n_wires - 1):
            tq.cx(q_device, wires=[w, w + 1])


class Kernel(tq.QuantumModule):
    """Quantum kernel that returns a fidelity‑based similarity."""

    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a scalar kernel value for two 1‑D feature vectors.

        Parameters
        ----------
        x, y
            Tensors of shape ``(features,)``.

        Returns
        -------
        kernel
            Scalar tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        self.ansatz(self.q_device, x, y)
        states = self.q_device.states.view(-1, 2 ** self.n_wires)
        # Fidelity with the |0...0> reference state
        fidelity = torch.abs(states[0, 0]) ** 2
        return fidelity


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of feature vectors.

    Parameters
    ----------
    a, b
        Sequences of 1‑D tensors.

    Returns
    -------
    gram
        NumPy array of shape ``(len(a), len(b))``.
    """
    kernel = Kernel()
    mat = np.zeros((len(a), len(b)))
    for i, ax in enumerate(a):
        for j, by in enumerate(b):
            mat[i, j] = kernel(ax, by).item()
    return mat
