"""
Quantum kernel implementation using TorchQuantum.

Provides an amplitude-encoded kernel with a parameterised ansatz.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torchquantum as tq
from torchquantum.functional import amplitude_encode
import numpy as np


class AmplitudeEncodedAnsatz(tq.QuantumModule):
    """
    Parameterised ansatz that applies Ry rotations followed by a ring of CNOTs.
    The ansatz is applied after amplitude encoding of the data.
    """

    def __init__(self, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
        # params shape: (depth, n_qubits)
        for d in range(self.depth):
            for q in range(self.n_qubits):
                tq.ry(q_device, wires=[q], params=params[d, q])
            # entangling layer
            for q in range(self.n_qubits):
                tq.cnot(q_device, wires=[q, (q + 1) % self.n_qubits])


class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that computes the overlap between the states
    produced by amplitude encoding of two input vectors followed by
    a parameterised ansatz.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.ansatz = AmplitudeEncodedAnsatz(n_qubits, depth)

    @tq.static_support
    def _encode(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        # Amplitude encode the data vector onto the qubits
        size = 2 ** self.n_qubits
        vec = x.squeeze().tolist()
        if len(vec) < size:
            vec += [0.0] * (size - len(vec))
        elif len(vec) > size:
            vec = vec[:size]
        vec = torch.tensor(vec, dtype=torch.complex64)
        amplitude_encode(q_device, vec)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value k(x, y).
        """
        self.q_device.reset_states(1)
        # Encode x
        self._encode(self.q_device, x)
        # Apply ansatz
        params_x = torch.randn(self.ansatz.depth, self.n_qubits)
        self.ansatz(self.q_device, params_x)
        # Save state for x
        state_x = self.q_device.states.clone()

        # Reset and encode y
        self.q_device.reset_states(1)
        self._encode(self.q_device, y)
        params_y = torch.randn(self.ansatz.depth, self.n_qubits)
        self.ansatz(self.q_device, params_y)
        # Compute overlap
        state_y = self.q_device.states
        overlap = torch.abs(torch.dot(state_x.squeeze(), state_y.squeeze()))
        return overlap


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    n_qubits: int = 4,
    depth: int = 2,
) -> np.ndarray:
    """
    Compute the Gram matrix between two datasets using the quantum kernel.
    """
    kernel = QuantumKernel(n_qubits, depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["AmplitudeEncodedAnsatz", "QuantumKernel", "kernel_matrix"]
