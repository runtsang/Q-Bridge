"""Quantum kernel using a variational circuit with Pennylane.

Features
--------
* Variational ansatz with trainable rotation angles.
* Data encoding via Ry rotations.
* Overlap measurement using state vectors (squared fidelity).
* GPU acceleration via the Lightning backend when available.
"""

from __future__ import annotations

import torch
import pennylane as qml
import pennylane.numpy as pnp
from typing import Sequence
import numpy as np


class QuantumKernelMethod:
    """
    Quantum kernel implemented with Pennylane.

    Parameters
    ----------
    num_qubits : int, optional
        Number of qubits / wires used for encoding.
    wires : Sequence[int] | None, optional
        Wire indices. Defaults to ``range(num_qubits)``.
    backend : str, optional
        Pennylane backend. Defaults to ``"default.qubit"``.
    device_kwargs : dict, optional
        Additional keyword arguments passed to ``qml.device``.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        wires: Sequence[int] | None = None,
        backend: str = "default.qubit",
        device_kwargs: dict | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.wires = wires or list(range(num_qubits))
        device_kwargs = device_kwargs or {}
        # Prefer Lightning GPU if available
        if backend == "default.qubit" and device_kwargs.get("device", "cpu") == "cuda":
            backend = "lightning.qubit"
            device_kwargs.setdefault("backend", "gpu")
        self.dev = qml.device(backend, wires=self.wires, **device_kwargs)
        # Trainable variational parameters
        self.params = torch.nn.Parameter(torch.randn(self.num_qubits, dtype=torch.float32))

        # Create a qnode that returns the state vector
        def _state_circuit(x: torch.Tensor) -> torch.Tensor:
            self._encoding(x)
            self._entanglement()
            self._variational_layer()
            return qml.state()

        self._state_qnode = qml.qnode(self.dev, interface="torch")(_state_circuit)

    def _encoding(self, x: torch.Tensor) -> None:
        """Encode classical data into qubits via Ry rotations."""
        for i, wire in enumerate(self.wires):
            qml.RY(x[i], wires=wire)

    def _entanglement(self) -> None:
        """Apply a simple entangling layer."""
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])

    def _variational_layer(self) -> None:
        """Apply a layer of trainable rotations."""
        for i, wire in enumerate(self.wires):
            qml.RY(self.params[i], wires=wire)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape (num_qubits,).

        Returns
        -------
        torch.Tensor
            Kernel value (squared fidelity) as a scalar tensor.
        """
        # Ensure tensors are on the correct device
        x = x.to(torch.get_default_dtype())
        y = y.to(torch.get_default_dtype())
        state_x = self._state_qnode(x)
        state_y = self._state_qnode(y)
        overlap = torch.abs(torch.dot(state_x.conj(), state_y)) ** 2
        return overlap

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Gram matrix between two sequences of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors of length ``num_qubits``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(a), len(b)).
        """
        A = torch.stack(a)
        B = torch.stack(b)
        m, n = A.shape[0], B.shape[0]
        K = torch.empty((m, n), dtype=torch.get_default_dtype())
        for i in range(m):
            for j in range(n):
                K[i, j] = self.forward(A[i], B[j])
        return K

    def to_numpy(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """
        Helper to compute the kernel matrix and return a NumPy array.

        Parameters
        ----------
        a, b : Sequence[np.ndarray]
            Sequences of 1‑D NumPy arrays of length ``num_qubits``.

        Returns
        -------
        np.ndarray
            Kernel matrix as a NumPy array.
        """
        torch_a = [torch.from_numpy(x).float() for x in a]
        torch_b = [torch.from_numpy(y).float() for y in b]
        return self.kernel_matrix(torch_a, torch_b).detach().cpu().numpy()


__all__ = ["QuantumKernelMethod"]
