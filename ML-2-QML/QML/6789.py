"""Quantum implementation of a kernel based on a feature‑map auto‑encoder.

The :class:`AutoKernel` class evaluates the overlap between two
quantum states that are prepared by a RealAmplitudes ansatz parameterised
by the classical input vectors.  The resulting value is a valid
positive‑definite kernel that can be used in quantum‑enhanced learning
algorithms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

class AutoKernel:
    """
    Quantum kernel using a RealAmplitudes feature map.

    Parameters
    ----------
    num_qubits : int
        Dimension of the input vectors (must match the number of qubits).
    reps : int, optional
        Number of repetitions in the RealAmplitudes ansatz.
    """
    def __init__(self, num_qubits: int, reps: int = 1) -> None:
        self.num_qubits = num_qubits
        self.reps = reps
        self.sim = AerSimulator(method='statevector')

    def _statevector(self, x: torch.Tensor) -> Statevector:
        """
        Prepare the quantum state for a single input vector.

        Parameters
        ----------
        x : torch.Tensor
            1‑D tensor of length ``num_qubits``.
        """
        qc = QuantumCircuit(self.num_qubits)
        if self.reps == 1:
            qc.append(RealAmplitudes(self.num_qubits, reps=1), range(self.num_qubits))
            qc.set_parameters(x.tolist())
        else:
            # Expand the input to match the number of parameters
            params = torch.cat([x] * self.reps).tolist()
            qc.append(RealAmplitudes(self.num_qubits, reps=self.reps), range(self.num_qubits))
            qc.set_parameters(params)
        result = self.sim.run(qc).result()
        return Statevector(result.get_statevector())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two batches of inputs.

        Parameters
        ----------
        x, y : torch.Tensor
            Tensors of shape (N, D) and (M, D) where D == num_qubits.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (N, M) with values in [0, 1].
        """
        n, d = x.shape
        m, d2 = y.shape
        if d!= self.num_qubits or d2!= self.num_qubits:
            raise ValueError("Input dimension must match num_qubits")
        kernel = torch.empty((n, m), dtype=torch.float32)
        for i in range(n):
            psi = self._statevector(x[i])
            psi_t = torch.from_numpy(psi.data).to(torch.complex64)
            for j in range(m):
                phi = self._statevector(y[j])
                phi_t = torch.from_numpy(phi.data).to(torch.complex64)
                overlap = torch.abs(torch.dot(psi_t.conj(), phi_t)) ** 2
                kernel[i, j] = overlap.item()
        return kernel

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  num_qubits: int,
                  reps: int = 1) -> np.ndarray:
    """
    Convenience wrapper to compute the Gram matrix for arbitrary
    sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors representing two datasets.
    num_qubits : int
        Number of qubits for the feature map.
    reps : int, optional
        Number of repetitions in the RealAmplitudes ansatz.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (len(a), len(b)).
    """
    kernel = AutoKernel(num_qubits, reps)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["AutoKernel", "kernel_matrix"]
