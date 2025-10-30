"""Variational quanvolution filter with measurement‑based attention."""

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer import AerSimulator
from qiskit import execute

class ConvEnhanced:
    """
    Variational quanvolution filter that supports multiple kernel sizes,
    trainable parameters, entangling layers, and a measurement‑based
    attention mask.  The API matches the classical version: the
    returned object has a ``run`` method that accepts a 2‑D array.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        backend: qiskit.providers.backend.Backend | None = None,
        shots: int = 1024,
        threshold: float = 0.5,
        depth: int = 3,
    ):
        self.kernel_sizes = kernel_sizes or [2]
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.depth = depth

        # Build a separate circuit for each kernel size
        self.circuits = []
        for k in self.kernel_sizes:
            n_qubits = k * k
            circ = QuantumCircuit(n_qubits)
            params = [Parameter(f"theta_{i}") for i in range(n_qubits)]
            for i, p in enumerate(params):
                circ.ry(p, i)
            for _ in range(self.depth):
                for i in range(n_qubits - 1):
                    circ.cx(i, i + 1)
                circ.barrier()
            circ.measure_all()
            self.circuits.append((circ, params))

    def run(self, data):
        """
        Execute the variational circuit on a single kernel‑sized patch.
        The data is a 2‑D array with values in [0,1].
        Returns a scalar between 0 and 1.
        """
        outputs = []
        for circ, params in self.circuits:
            k = int(np.sqrt(circ.num_qubits))
            vec = np.reshape(data, (k * k,))
            bind = {p: np.pi if v > self.threshold else 0 for p, v in zip(params, vec)}
            job = execute(circ, self.backend, shots=self.shots, parameter_binds=[bind])
            result = job.result()
            counts = result.get_counts(circ)

            # Compute expectation of Z for each qubit
            exp_vals = []
            for i in range(circ.num_qubits):
                ones = sum(int(state[i]) * cnt for state, cnt in counts.items())
                exp = (self.shots - 2 * ones) / self.shots  # <Z>
                exp_vals.append(exp)

            # Attention mask: use absolute expectation values
            attention = np.mean(np.abs(exp_vals))
            out = np.mean(np.abs(exp_vals)) * attention
            outputs.append(out)

        return np.mean(outputs)

def Conv() -> ConvEnhanced:
    """Return a ConvEnhanced instance that mimics the original API."""
    return ConvEnhanced()

__all__ = ["ConvEnhanced"]
