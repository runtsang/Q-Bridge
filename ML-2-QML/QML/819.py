"""Quantum‑enhanced convolutional filter using a parameter‑shift ansatz.

This module implements a variational circuit that mimics the behaviour of a classical
convolutional filter but allows for gradient‑friendly training via the parameter‑shift
rule. The circuit is built on top of Qiskit and can be executed on any Aer simulator
or real backend.

The class shares the same name as the classical counterpart (ConvEnhanced) to
facilitate hybrid usage.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import Parameter
from typing import Tuple

class ConvEnhanced:
    """
    Variational quanvolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (assumes square kernel).
    shots : int
        Number of shots for each execution.
    threshold : float
        Threshold for encoding classical data into rotation angles.
    backend : qiskit.providers.backend.Backend
        Quantum backend to execute the circuit.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 1024,
                 threshold: float = 127.0,
                 backend=None) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        self.n_qubits = kernel_size ** 2
        self.theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]

        # Build a simple parameter‑shift ansatz
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def encode_data(self, data: np.ndarray) -> dict:
        """
        Encode classical data into rotation angles.
        Values above the threshold are mapped to π, otherwise to 0.
        """
        flat = data.flatten()
        binds = {}
        for i, val in enumerate(flat):
            binds[self.theta[i]] = np.pi if val > self.threshold else 0.0
        return binds

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on classical data and return the average
        probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(f"Expected data shape {(self.kernel_size, self.kernel_size)}, got {data.shape}")

        binds = self.encode_data(data)
        bound_circ = self.circuit.bind_parameters(binds)
        transpiled = transpile(bound_circ, self.backend)
        qobj = assemble(transpiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts(bound_circ)

        # Compute average probability of |1> over all qubits
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * freq
        prob = total_ones / (self.shots * self.n_qubits)
        return prob

    def predict(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Return probability and log‑likelihood for a single input.
        """
        prob = self.run(data)
        log_likelihood = np.log(prob + 1e-12)
        return prob, log_likelihood
