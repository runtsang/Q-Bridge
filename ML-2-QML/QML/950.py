"""SharedKernel: quantum kernel based on a variational circuit.

The implementation uses PennyLane to evaluate the overlap of two encoded
states.  A simple hardware‑efficient ansatz is employed, but the class
is designed to accept any custom circuit via the ``circuit`` argument.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn


class SharedKernel(nn.Module):
    """Quantum kernel module.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits in the circuit.
    device : str, optional
        PennyLane device name (default: ``default.qubit``).
    circuit : Callable[[torch.Tensor, Sequence[int]], None], optional
        User‑supplied function that applies gates to the device.
        If ``None`` a default RX‑RZ‑CNOT ansatz is used.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 device: str = "default.qubit",
                 circuit: None | Callable[[torch.Tensor, Sequence[int]], None] = None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self.circuit = circuit or self._default_ansatz

        @qml.qnode(self.dev, interface="torch")
        def _qnode(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode x
            self.circuit(x, wires=range(n_qubits))
            # Apply inverse of y‑encoding by negating the parameters
            self.circuit(-y, wires=range(n_qubits))
            # Return probability of the all‑zero state
            return qml.probs(wires=range(n_qubits))[0]

        self._qnode = _qnode

    def _default_ansatz(self, params: torch.Tensor, wires: Sequence[int]) -> None:
        """Default hardware‑efficient ansatz: RX followed by RZ on each qubit,
        then a layer of CNOTs in a ring topology."""
        for w in wires:
            qml.RX(params[w], wires=w)
            qml.RZ(params[w], wires=w)
        for w in wires:
            qml.CNOT(wires=[w, wires[(w + 1) % len(wires)]])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``k(x, y)`` as the overlap probability."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return torch.abs(self._qnode(x, y))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix for two batches of samples."""
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
        K = torch.empty(a.shape[0], b.shape[0])
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                K[i, j] = self.forward(xi, yj)
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_qubits: int = 4) -> np.ndarray:
    """Convenience wrapper that builds a :class:`SharedKernel` and returns a NumPy array."""
    kernel = SharedKernel(n_qubits=n_qubits)
    return kernel.kernel_matrix(torch.stack(a), torch.stack(b)).detach().cpu().numpy()


__all__ = ["SharedKernel", "kernel_matrix"]
