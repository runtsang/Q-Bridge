"""Quantum kernel construction using Pennylane."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as pq
from pennylane import numpy as pnp


class KernalAnsatz:
    """
    Parameter‑encoded ansatz that applies RX rotations followed by a linear
    entanglement chain.  The ansatz is symmetric: it first encodes `x`,
    then `y` with a negative sign, enabling overlap estimation.
    """

    def __init__(self, n_qubits: int = 4, entanglement: str = "cnot") -> None:
        self.n_qubits = n_qubits
        self.entanglement = entanglement

    def _apply_rotations(self, dev: pq.Device, params: torch.Tensor) -> None:
        """Apply RX rotations to each qubit."""
        for i in range(self.n_qubits):
            dev.apply(pq.ops.RX(params[i].item(), wires=i))

    def _apply_entanglement(self, dev: pq.Device) -> None:
        """Apply a linear CNOT chain."""
        if self.entanglement == "cnot":
            for i in range(self.n_qubits - 1):
                dev.apply(pq.ops.CNOT(wires=[i, i + 1]))
        else:
            raise ValueError(f"Unsupported entanglement: {self.entanglement}")

    def forward(
        self, dev: pq.Device, x: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Encode two data vectors into the quantum state.

        Parameters
        ----------
        dev : pennylane.Device
            Quantum device to run the circuit.
        x, y : torch.Tensor
            1‑D tensors of length `n_qubits`.
        """
        dev.reset()
        self._apply_rotations(dev, x)
        self._apply_entanglement(dev)
        self._apply_rotations(dev, -y)
        self._apply_entanglement(dev)


class Kernel:
    """
    Quantum kernel module that exposes a classical interface.
    """

    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = pq.Device(dev_name, wires=n_qubits)
        self.ansatz = KernalAnsatz(n_qubits)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel for a single pair.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of length `n_qubits`.

        Returns
        -------
        torch.Tensor
            Overlap value |⟨ψ(x)|ψ(y)⟩|² as a float tensor.
        """
        self.ansatz.forward(self.dev, x, y)
        state = self.dev.state
        # Compute the probability of the |0…0⟩ state
        overlap = abs(state[0]) ** 2
        return torch.tensor(overlap, dtype=torch.float32)

    def forward_batch(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch evaluation of the quantum kernel.

        Parameters
        ----------
        X, Y : torch.Tensor of shape (n, n_qubits) and (m, n_qubits)

        Returns
        -------
        torch.Tensor of shape (n, m)
            Pairwise kernel matrix.
        """
        n, m = X.shape[0], Y.shape[0]
        result = torch.empty((n, m), dtype=torch.float32)
        for i in range(n):
            for j in range(m):
                result[i, j] = self.forward(X[i], Y[j])
        return result


def kernel_matrix(
    a: Sequence[torch.Tensor | np.ndarray],
    b: Sequence[torch.Tensor | np.ndarray],
) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of points using the
    quantum kernel.

    Parameters
    ----------
    a, b : sequence of tensors or numpy arrays
        Data points to evaluate.

    Returns
    -------
    np.ndarray
        Pairwise kernel matrix.
    """
    kernel = Kernel()
    X = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in a])
    Y = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in b])
    return kernel.forward_batch(X, Y).detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
