"""
ConvEnhanced: Quantum filter that produces an attention mask for a 2×2 patch.
The circuit encodes each pixel as a rotation angle, entangles the qubits,
and returns the expectation value of Pauli‑Z on the first qubit as a mask
in [0, 1]. This mask can be used to modulate a classical convolution output.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class _ConvEnhancedQuantum:
    """
    Variational quantum filter for a 2×2 convolutional kernel.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 1024):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')

        # Parameters for data encoding
        self.theta = [Parameter(f'theta{i}') for i in range(self.n_qubits)]

        # Build a simple variational circuit
        self._circuit = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(self.theta):
            self._circuit.rx(p, i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)
        # Measurement of all qubits
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns:
            float: mask value in [0, 1] derived from the circuit expectation.
        """
        flat = data.reshape(self.n_qubits)
        # Bind each theta to a rotation angle proportional to pixel intensity
        param_binds = {self.theta[i]: np.pi * flat[i] / 255.0 for i in range(self.n_qubits)}
        job = execute(self._circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute expectation value of Pauli‑Z on qubit 0
        exp = 0.0
        for bitstring, cnt in counts.items():
            # bitstring is ordered with qubit 0 as the last bit
            val = 1 if bitstring[-1] == '1' else -1
            exp += val * cnt
        exp = exp / (self.shots * self.n_qubits)

        # Map expectation to [0, 1]
        return (exp + 1.0) / 2.0

def ConvEnhanced():
    """
    Factory returning a ConvEnhancedQuantum instance.
    """
    return _ConvEnhancedQuantum()

__all__ = ["ConvEnhanced"]
