"""Quantum convolution module that implements a quanvolution filter.

The original Conv class used a simple random circuit.  ConvGen264
adds a learnable threshold and exposes a `run` method compatible
with the classical API.  The public `Conv()` function returns an
instance of this class.
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class ConvGen264:
    """Quantum implementation of a convolutional filter.

    The filter processes a 2‑D kernel of shape (kernel_size, kernel_size)
    by encoding each pixel as a rotation angle on a qubit and running
    a parameterised circuit.  The output is the average probability
    of measuring |1> across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 127.0,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build the circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """Execute the circuit on a single kernel.

        Args:
            data: 2‑D array or list of shape (kernel_size, kernel_size).

        Returns:
            Average probability of measuring |1> across qubits.
        """
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        data = data.reshape(1, self.n_qubits)

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

    def predict(self, data: np.ndarray | list[list[float]]) -> float:
        """Convenience wrapper that mimics the classical API."""
        return self.run(data)

def Conv() -> ConvGen264:
    """Convenience factory that mimics the original API."""
    return ConvGen264()
