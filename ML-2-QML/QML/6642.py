"""Quantum kernel construction with a parameterised ansatz and amplitude encoding.

The :class:`QuantumKernelMethod` is a :class:`torchquantum.QuantumModule` that
prepares two amplitude‑encoded states, applies a learnable variational circuit,
and returns the magnitude of the overlap.  It also exposes a ``kernel_matrix``
utility that operates on sequences of tensors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torch import nn
from torchquantum.functional import amplitude_encode


class QuantumKernelMethod(tq.QuantumModule):
    """Quantum RBF‑like kernel based on amplitude encoding and a trainable ansatz.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits used for each data vector.  The data must have
        ``2**n_wires`` components, which are interpreted as amplitudes.
    depth : int, default=2
        Depth of the variational circuit.  Each layer consists of single‑qubit
        rotations followed by a chain of CNOTs.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable parameters: (depth, n_wires, 2) for RY and RZ angles
        self.params = nn.Parameter(torch.randn(depth, n_wires, 2))

    def _apply_ansatz(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
        """Apply a single layer of the variational circuit."""
        for d in range(self.depth):
            for w in range(self.n_wires):
                tq.functional.ry(q_device, wires=[w], params=params[d, w, 0])
                tq.functional.rz(q_device, wires=[w], params=params[d, w, 1])
            if d < self.depth - 1:
                for w in range(self.n_wires - 1):
                    tq.functional.cnot(q_device, wires=[w, w + 1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two amplitude‑encoded vectors.

        The data vectors must have length ``2**self.n_wires``.  The kernel is
        defined as the magnitude of the overlap between the two variationally
        transformed states.
        """
        if x.shape[-1]!= 2 ** self.n_wires or y.shape[-1]!= 2 ** self.n_wires:
            raise ValueError("Data dimension must be 2**n_wires.")
        self.q_device.reset_states(1)
        # Encode the first vector
        amplitude_encode(self.q_device, x)
        self._apply_ansatz(self.q_device, self.params)
        # Encode the second vector and apply the inverse circuit
        amplitude_encode(self.q_device, y)
        self._apply_ansatz(self.q_device, -self.params)
        # The overlap is the absolute value of the amplitude of the |0> state
        return torch.abs(self.q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper that returns a NumPy Gram matrix."""
        model = QuantumKernelMethod()
        mat = torch.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = model.forward(x, y)
        return mat.cpu().numpy()


__all__ = ["QuantumKernelMethod"]
