"""Quantum kernel implementation using Pennylane.

The class mirrors the classical :class:`QuantumKernelMethod` API
while evaluating a quantum kernel via a parameterized circuit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(nn.Module):
    """A quantum kernel wrapper that encodes data into a variational circuit.

    The implementation uses Pennylane's :class:`~pennylane.Device` to
    construct a parameterized ansatz.  The kernel is defined as the
    squared fidelity between the states produced by encoding ``x`` and
    ``y``.
    """

    def __init__(
        self,
        wires: int = 4,
        dev: qml.Device | None = None,
        ansatz: str = "ry",
        layers: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        wires : int
            Number of qubits used to encode the data.
        dev : pennylane.Device, optional
            Pennylane device.  If ``None`` a default ``default.qubit`` device
            is created.
        ansatz : str, optional
            Single‑qubit rotation used in the variational layers.  Supported
            values are ``"ry"`` and ``"rz"``.
        layers : int, optional
            Number of variational layers.
        device : str or torch.device, optional
            Device on which to perform the classical tensor operations.
        """
        super().__init__()
        self.wires = wires
        self.layers = layers
        self.device = torch.device(device)

        if dev is None:
            self.dev = qml.device("default.qubit", wires=self.wires)
        else:
            self.dev = dev

        if ansatz not in {"ry", "rz"}:
            raise ValueError(f"Unsupported ansatz: {ansatz!r}")
        self.ansatz = ansatz

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(
            0.01 * torch.randn(layers, wires, requires_grad=True)
        )

    def _encode(self, data: torch.Tensor) -> None:
        """Encode a batch of data onto the qubits via rotation gates."""
        for i, val in enumerate(data):
            if self.ansatz == "ry":
                qml.RY(val, wires=i)
            else:
                qml.RZ(val, wires=i)

    def _variational_layer(self, params: torch.Tensor) -> None:
        """Apply a single variational layer."""
        for i in range(self.wires):
            qml.RY(params[0, i], wires=i)
        for i in range(self.wires - 1):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(self.dev, interface="torch")
    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Quantum kernel circuit.  Computes the absolute value of the overlap."""
        self._encode(x)
        self._variational_layer(self.params)
        self._encode(-y)  # reverse encoding of y
        self._variational_layer(self.params)
        return torch.abs(qml.state()[0]) ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value ``k(x, y)``.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape ``(wires,)``.  The tensors are moved to
            ``self.device`` and cast to ``torch.float64`` for Pennylane.
        """
        x = x.to(self.device).double()
        y = y.to(self.device).double()
        return self._quantum_kernel(x, y)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two batches of samples.

        Parameters
        ----------
        a, b : torch.Tensor
            Batches of shape ``(n, wires)`` and ``(m, wires)`` respectively.
        """
        n, m = a.shape[0], b.shape[0]
        gram = torch.empty(n, m, device=self.device, dtype=torch.float64)
        for i in range(n):
            for j in range(m):
                gram[i, j] = self(a[i], b[j])
        return gram

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Utility to convert a torch tensor to a numpy array."""
        return tensor.detach().cpu().numpy()
