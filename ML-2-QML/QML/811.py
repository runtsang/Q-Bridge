"""Quantum kernel construction using PennyLane with a parameterised circuit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml


class KernalAnsatz:
    """Quantum kernel ansatz that encodes two classical vectors and returns the
    overlap with the computational basis state |0...0⟩.

    Parameters
    ----------
    wires : Sequence[int]
        Wire indices on which the circuit acts.
    """

    def __init__(self, wires: Sequence[int]) -> None:
        self.wires = wires
        dev = qml.device("default.qubit", wires=len(wires))
        self.qnode = qml.QNode(self._circuit, dev, interface="torch")

    def _circuit(self, x: torch.Tensor, y: torch.Tensor):
        # Encode x
        for i, w in enumerate(self.wires):
            qml.RY(x[i], wires=w)
        # Fixed entangling layer
        for i in range(len(self.wires) - 1):
            qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
        # Unencode y with negative parameters
        for i, w in enumerate(self.wires):
            qml.RY(-y[i], wires=w)
        return qml.state()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        state = self.qnode(x, y)
        # Overlap with |0...0⟩ is the first amplitude
        return torch.abs(state[0]) ** 2


class Kernel:
    """Convenience wrapper exposing a two‑argument call interface.

    The wrapper caches the ansatz and avoids re‑instantiating the quantum
    device on every call.
    """

    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.ansatz = KernalAnsatz(wires=list(range(n_wires)))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix using the quantum kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors that can be stacked into shape ``(n, d)``.
    Returns
    -------
    numpy.ndarray
        The kernel matrix of shape ``(len(a), len(b))``.
    """
    a = [torch.as_tensor(x, dtype=torch.float32) for x in a]
    b = [torch.as_tensor(y, dtype=torch.float32) for y in b]
    mat = torch.stack([torch.stack([Kernel()(x, y) for y in b]) for x in a])
    return mat.detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
