"""Quantum kernel using a variational circuit on Pennylane.

The implementation is fully compatible with the classical counterpart and
provides an end‑to‑end quantum kernel that can be trained with a classical
optimizer.  The kernel is evaluated on a statevector simulator for exact
results, but can be switched to a shot‑based device for NISQ deployment.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from typing import Sequence

class KernalAnsatz:
    """Variational ansatz for data encoding and feature map.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the circuit.
    depth : int
        Number of repeated RY→CNOT layers.
    seed : int, optional
        Random seed for weight initialization.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2, seed: int | None = None) -> None:
        self.n_wires = n_wires
        self.depth = depth
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(0, 2 * np.pi, size=(depth, n_wires))

    def __call__(self, dev: qml.Device, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply the encoding and inverse encoding on the quantum device."""
        # encode x
        for idx, wire in enumerate(range(self.n_wires)):
            dev.apply(qml.RY(x[0, idx], wires=wire))
        # entangle with layers
        for _ in range(self.depth):
            for wire in range(self.n_wires - 1):
                dev.apply(qml.CNOT(wires=[wire, wire + 1]))
            for wire in range(self.n_wires):
                dev.apply(qml.RY(self.params[0, wire], wires=wire))
        # encode -y (inverse)
        for idx, wire in enumerate(range(self.n_wires)):
            dev.apply(qml.RY(-y[0, idx], wires=wire))

class Kernel:
    """Quantum kernel that evaluates the overlap of two encoded states.

    The kernel is defined as::

        k(x, y) = |⟨ψ(x)|ψ(y)⟩|²

    where ``ψ(x)`` is the state prepared by :class:`KernalAnsatz`.
    """

    def __init__(self, n_wires: int = 4, depth: int = 2,
                 backend: str = "default.qubit", shots: int | None = None) -> None:
        self.n_wires = n_wires
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.ansatz = KernalAnsatz(n_wires, depth)

        # Device for exact statevector evaluation
        self.dev = qml.device(self.backend, wires=self.n_wires,
                              shots=shots)

        @qml.qnode(self.dev, interface="torch")
        def _statevector(x):
            self.ansatz(self.dev, x, torch.zeros_like(x))
            return qml.state()

        self._statevector = _statevector

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for two 1‑D tensors."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        psi_x = self._statevector(x)
        psi_y = self._statevector(y)
        return torch.abs(torch.vdot(psi_x, psi_y)) ** 2

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        A = torch.stack(a).float()
        B = torch.stack(b).float()
        K = torch.stack([self.__call__(a_i, b_i) for a_i in A for b_i in B])
        return K.reshape(len(a), len(b)).detach().cpu().numpy()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Convenience wrapper that instantiates a default Kernel."""
    kernel = Kernel()
    return kernel.kernel_matrix(a, b)

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
