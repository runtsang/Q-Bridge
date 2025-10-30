"""Quantum kernel implemented with PennyLane's variational ansatz."""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import torch
import pennylane as qml

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod:
    """Hybrid kernel that uses a PennyLane variational circuit.

    The ansatz encodes each input vector via RY rotations and
    subsequently applies two layers of Hadamards and CNOTs.  The kernel
    value is the absolute overlap of the two resulting quantum states.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits (and input dimensions) in the circuit.
    wires : Iterable[int], optional
        Wire indices used in the PennyLane device.  Defaults to
        ``range(n_qubits)``.
    device : str, optional
        PennyLane backend (e.g. ``'default.qubit'`` or ``'qiskit.ibmq'``).

    Notes
    -----
    The class is intentionally lightweight; the circuit is compiled
    once during construction and reused for all kernel evaluations.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 wires: Iterable[int] | None = None,
                 device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.wires = list(wires) if wires is not None else list(range(n_qubits))
        self.dev = qml.device(device, wires=self.wires)
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(data: torch.Tensor):
            # data shape: (n_qubits,)
            for i, wire in enumerate(self.wires):
                qml.RY(data[i], wires=wire)
            # Two trainable layers
            for _ in range(2):
                for wire in self.wires:
                    qml.Hadamard(wire)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            return qml.state()
        self.circuit = circuit

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute inner product of the two encoded states."""
        psi_x = self.circuit(x)
        psi_y = self.circuit(y)
        return torch.abs(torch.vdot(psi_x, psi_y))

def kernel_matrix(a: Sequence[Union[np.ndarray, torch.Tensor]],
                  b: Sequence[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
    """Compute the Gram matrix for a list of data points.

    Parameters
    ----------
    a, b : sequences of tensors or arrays
        Input data; each element must be a vector of length equal to
        the circuit's number of qubits.

    Returns
    -------
    np.ndarray
        The kernel matrix ``K_{ij} = k(a_i, b_j)``.
    """
    qkernel = QuantumKernelMethod()
    a_t = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in a]
    b_t = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in b]
    matrix = np.array([[qkernel.kernel(x, y).item() for y in b_t] for x in a_t])
    return matrix
