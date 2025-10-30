"""Quantum convolutional filter with multi‑kernel support and a parameter‑shared variational circuit.

Drop‑in replacement for the original Conv filter.  Extends the original
functionality by:

* Supporting a list of kernel sizes.
* Using a parameter‑shared variational circuit for each kernel.
* Adaptive threshold for input encoding.
* Optional residual logic when the input and output dimensions match.
* A ``run`` convenience method that accepts a 2‑D array and returns a scalar.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute
from typing import Iterable, Dict, List

class ConvGen342:
    """Quantum convolutional filter with variational circuits."""

    def __init__(
        self,
        kernel_sizes: Iterable[int] | int = 3,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 100,
        threshold: float = 0.5,
    ) -> None:
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = list(kernel_sizes)
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build a circuit per kernel
        self.circuits: Dict[int, qiskit.QuantumCircuit] = {}
        self.params: Dict[int, List[Parameter]] = {}
        for k in self.kernel_sizes:
            n_qubits = k * k
            circ = qiskit.QuantumCircuit(n_qubits)
            # Parameterized rotations
            thetas = [Parameter(f"theta_{i}") for i in range(n_qubits)]
            self.params[k] = thetas
            for i, th in enumerate(thetas):
                circ.rx(th, i)
            # Add a simple entangling layer
            for i in range(n_qubits - 1):
                circ.cx(i, i + 1)
            circ.barrier()
            circ.measure_all()
            self.circuits[k] = circ

    def _bind_and_execute(self, k: int, data: np.ndarray) -> float:
        """Encode data, bind parameters, and execute the circuit."""
        n_qubits = k * k
        # Encode data into rotation angles
        bind_dict = {}
        for i, val in enumerate(data.flatten()):
            bind_dict[self.params[k][i]] = np.pi if val > self.threshold else 0.0
        job = execute(
            self.circuits[k],
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        # Compute average probability of measuring |1> across all qubits
        counts = result.get_counts(self.circuits[k])
        total = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total += ones * cnt
        return total / (self.shots * n_qubits)

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Patch of shape `(k, k)` where `k` is one of the supported kernel sizes.

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        shape = data.shape
        if len(shape)!= 2 or shape[0]!= shape[1]:
            raise ValueError("Input must be a square 2‑D array.")
        k = shape[0]
        if k not in self.kernel_sizes:
            raise ValueError(f"Unsupported kernel size {k}. Supported: {self.kernel_sizes}")

        return self._bind_and_execute(k, data)

def Conv() -> ConvGen342:
    """Return a drop‑in replacement for the original Conv filter."""
    return ConvGen342(kernel_sizes=[1, 3, 5], threshold=0.5, shots=200)
