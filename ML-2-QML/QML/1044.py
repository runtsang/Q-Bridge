"""Quantum convolutional filter using a parameter‑shared variational circuit.

The ConvEnhancedQuantum class implements a variational circuit that mimics
the behaviour of the classical ConvEnhanced filter.  Each qubit corresponds
to one pixel of a kernel and receives a rotation that depends on the pixel
value relative to a threshold.  A single layer of CNOT entanglement is
applied, followed by a second layer of rotations that share parameters
across qubits.  The output is the average probability of measuring |1>
across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from qiskit.circuit import Parameter
from typing import Optional


class ConvEnhancedQuantum:
    """Variational quantum filter for 2‑D kernels.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (e.g., 2 for a 2×2 filter).
    threshold : float, default=0.5
        Threshold used to decide the rotation angle for each qubit.
    shots : int, default=1024
        Number of shots for the backend execution.
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit.  If None, the Aer qasm simulator
        is used.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        shots: int = 1024,
        backend: Optional[qiskit.providers.Backend] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # build the circuit
        self._circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        # first rotation layer
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta[i], i)

        # entanglement layer (chain of CNOTs)
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)

        # second rotation layer with shared parameters
        shared_theta = Parameter("shared_theta")
        for i in range(self.n_qubits):
            self._circuit.ry(shared_theta, i)

        # measurement
        self._circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single kernel.

        Parameters
        ----------
        data : array-like, shape (kernel_size, kernel_size)
            Classical pixel values in the range [0, 1].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        if data.ndim!= 2 or data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError("data must be a square array of shape "
                             f"({self.kernel_size}, {self.kernel_size})")

        # flatten data
        flat = data.flatten()
        # bind parameters based on threshold
        param_binds = []
        for val in flat:
            angle = np.pi / 2 if val > self.threshold else 0.0
            param_binds.append({self.theta[i]: angle for i in range(self.n_qubits)})

        # second layer shared parameter set to zero (no additional effect)
        bind_shared = {self.theta[i]: 0.0 for i in range(self.n_qubits)}
        bind_shared.update({self.theta[i]: 0.0 for i in range(self.n_qubits)})

        # combine binds
        binds = [dict(b, **bind_shared) for b in param_binds]

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # compute average probability of |1>
        total_ones = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
        prob = total_ones / (self.shots * self.n_qubits)
        return prob

    def __repr__(self) -> str:
        return (f"ConvEnhancedQuantum(kernel_size={self.kernel_size}, "
                f"threshold={self.threshold}, shots={self.shots})")


__all__ = ["ConvEnhancedQuantum"]
