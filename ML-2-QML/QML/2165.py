"""Quantum kernel using PennyLane with entanglement and variational ansatz."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import pennylane as qml
import torch


class QuantumKernel:
    """Quantum kernel evaluated via a PennyLane variational circuit.

    The ansatz consists of a layer of data‑encoding RY rotations followed
    by a chain of CNOTs to introduce entanglement.  The kernel value is
    the absolute overlap of the resulting state vectors.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits / circuit depth. Default is 4.
    dev : Optional[pennylane.Device], optional
        PennyLane device; if ``None`` the default ``default.qubit`` is used.
    """

    def __init__(self, n_qubits: int = 4, dev: Optional[qml.Device] = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def _state_circuit(x: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self._state_circuit = _state_circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``|⟨ψ(x)|ψ(y)⟩|``.

        Parameters
        ----------
        x, y : torch.Tensor
            Feature vectors of shape ``(n,)`` or ``(n, d)`` where ``d`` equals
            ``n_qubits``.  Broadcasting is not supported; callers should
            reshape appropriately.
        """
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        state_x = self._state_circuit(x)
        state_y = self._state_circuit(y)

        return torch.abs(torch.vdot(state_x, state_y))

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute the Gram matrix for two batches of feature vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of vectors with dimensionality equal to ``n_qubits``.
        """
        A = torch.stack([torch.tensor(v, dtype=torch.float32) for v in a])
        B = torch.stack([torch.tensor(v, dtype=torch.float32) for v in b])

        def kernel_row(x):
            return torch.stack([self(x, y) for y in B])

        return torch.stack([kernel_row(x) for x in A]).cpu().numpy()


__all__ = ["QuantumKernel"]
