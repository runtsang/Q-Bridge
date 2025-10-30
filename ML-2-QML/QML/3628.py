"""Hybrid kernel method with quantum kernel and a quantum fully‑connected layer.

The implementation follows the structure of the classical module but
replaces the RBF kernel with a parameterised quantum circuit implemented
in Qiskit.  The fully‑connected layer is also a small quantum circuit
that returns the expectation of a single qubit.

Key design choices
------------------
* **Quantum kernel** – A 4‑qubit ansatz that encodes two input vectors
  sequentially using Ry rotations.  The overlap of the final state with
  the initial state is used as the kernel value.
* **Quantum fully‑connected layer** – A 1‑qubit circuit that applies an
  Ry gate parametrised by the input and measures the expectation of Z.
  This mirrors the behaviour of the classical layer but uses a quantum
  backend.
* **Compatibility** – The public API matches the classical module
  (`kernel_matrix` and `apply_fcl`).  The class can be dropped in
  place of the classical one in existing pipelines.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter


class QuantumFCL:
    """Quantum implementation of the fully‑connected layer."""

    def __init__(self, backend=None, shots: int = 100) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the quantum circuit for a list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of parameters for the Ry gate.

        Returns
        -------
        np.ndarray
            Expectation value of the measured qubit (Z expectation),
            wrapped in a NumPy array to match the classical API.
        """
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(thetas, 0)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        expectation = 0.0
        for state, cnt in result.items():
            prob = cnt / self.shots
            expectation += float(state) * prob
        return np.array([expectation])


class HybridKernelMethod:
    """Quantum‑enhanced kernel method with a quantum fully‑connected layer."""

    def __init__(self, n_qubits: int = 4, shots: int = 1024, backend=None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.fcl = QuantumFCL(self.backend, self.shots)

    # ------------------------------------------------------------------ #
    # Quantum kernel utilities
    # ------------------------------------------------------------------ #
    def _kernel_circuit(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Build and execute a circuit that computes the kernel value
        between two vectors using Ry rotations.

        Parameters
        ----------
        x, y : Sequence[float]
            Input feature vectors to be encoded.

        Returns
        -------
        float
            Kernel value estimated from measurement statistics.
        """
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        for i, val in enumerate(x):
            qc.ry(val, i)
        qc.barrier()
        for i, val in enumerate(y):
            qc.ry(-val, i)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        counts = job.result().get_counts(qc)
        expectation = 0.0
        for state, cnt in counts.items():
            prob = cnt / self.shots
            expectation += float(state) * prob
        return expectation

    def kernel_matrix(
        self,
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Compute the Gram matrix between two datasets using the quantum kernel.

        Parameters
        ----------
        a, b : Sequence[Sequence[float]]
            Datasets represented as iterable of feature vectors.

        Returns
        -------
        np.ndarray
            Gram matrix of shape ``(len(a), len(b))``.
        """
        return np.array(
            [[self._kernel_circuit(x, y) for y in b] for x in a],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    # Quantum fully‑connected layer utility
    # ------------------------------------------------------------------ #
    def apply_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Delegate to the quantum fully‑connected layer.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters for the quantum circuit.

        Returns
        -------
        np.ndarray
            Output of the quantum fully‑connected layer.
        """
        return self.fcl.run(thetas)

__all__ = ["HybridKernelMethod", "QuantumFCL"]
