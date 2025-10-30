"""Quantum primitives for the hybrid autoencoder.

This module provides a variational quantum encoder that maps a classical
latent vector to a new latent representation consisting of expectation
values of Pauli‑Z on a set of wires.  It also implements a simple
overlap‑based quantum kernel that can be used for regularisation or
downstream similarity queries.

The implementation uses Pennylane because it exposes a clean autograd
interface and is lightweight for simulation.  The design mirrors the
structure of the classical RBF kernel in the original reference,
but replaces the Gaussian kernel with a quantum fidelity estimate.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp


class QuantumEncoder:
    """Variational quantum circuit that transforms a classical vector.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, equal to the target dimensionality of the
        quantum latent vector.
    """
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(params: np.ndarray) -> np.ndarray:
            # encode each dimension with a Ry rotation
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # entangle neighbours in a linear chain
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def __call__(self, params: np.ndarray | torch.Tensor) -> np.ndarray:
        """Run the circuit and return a NumPy array."""
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()
        return self.circuit(params)


class QuantumKernel:
    """Overlap‑based kernel between two quantum states.

    The kernel is defined as the absolute value of the inner product of the
    expectation‑value vectors produced by :class:`QuantumEncoder`.
    """
    def __init__(self, n_qubits: int) -> None:
        self.encoder = QuantumEncoder(n_qubits)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value as a scalar torch tensor."""
        # Ensure 1‑D tensors
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        psi_x = self.encoder(x_np)
        psi_y = self.encoder(y_np)
        # Compute absolute inner product
        val = np.abs(np.dot(psi_x, psi_y))
        return torch.tensor(val, dtype=torch.float32)

    def kernel_matrix(
        self,
        a: Iterable[torch.Tensor],
        b: Iterable[torch.Tensor],
    ) -> np.ndarray:
        """Compute the Gram matrix between two sets of vectors."""
        return np.array(
            [[self.kernel(x, y).item() for y in b]
             for x in a]
        )


__all__ = ["QuantumEncoder", "QuantumKernel"]
