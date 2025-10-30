"""Quantum convolution filter.

This module defines ConvGen, a quantum implementation of the original Conv filter.
It builds a parameter‑shared variational circuit that outputs a probability map
over qubits. The circuit can be executed on a Qiskit simulator or a real backend.

Typical usage::

    >>> from ConvGen import Conv
    >>> model = Conv(kernel_size=2, shots=200, threshold=0.5)
    >>> out = model.run(np.random.randint(0, 256, size=(2,2)))
    >>> print(out)
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer, execute


class ConvGen:
    """Quantum convolution filter."""
    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 0.5,
        backend=None,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend('qasm_simulator')
        # Build a simple variational circuit
        self.theta = [Parameter(f'theta_{i}') for i in range(self.n_qubits)]
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += RealAmplitudes(self.n_qubits, reps=1, entanglement='full')
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on the given 2‑D data.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with integer values.

        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (self.n_qubits,))
        param_binds = []
        for val in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i in range(self.n_qubits)}
            param_binds.append(bind)
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * cnt
            total_counts += cnt
        return total_ones / (total_counts * self.n_qubits)


def Conv(*args, **kwargs):
    """Factory that returns a ConvGen instance."""
    return ConvGen(*args, **kwargs)
