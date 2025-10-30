"""Hybrid quantum convolutional regression module.

It combines the quanvolution filter from the QML seed with an
EstimatorQNN‑style variational circuit.  The circuit is built with
Qiskit, and the expectation value of a Pauli‑Z observable is
returned as the regression output.  The module can be used as a
drop‑in replacement for the classical filter in a quantum‑aware
pipeline.
"""

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import torch
from torch import nn


class HybridQuantumFilter:
    """
    Quantum filter that encodes a 2×2 patch into a 4‑qubit state,
    applies a random variational layer, and measures the expectation
    of a Pauli‑Z observable.  The result is passed through a
    classical linear head to produce a scalar output.
    """
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100,
                 threshold: float = 127.0):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self.shots = shots

        # Build parameterised circuit
        self.input_params = [Parameter(f"in{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)
        # Random variational layer
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])

        # Estimator for expectation values
        self.estimator = StatevectorEstimator()
        # Classical linear head
        self.linear = nn.Linear(1, 1)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a 2×2 patch and return a regression output.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(f"Expected shape {(self.kernel_size, self.kernel_size)}")
        # Flatten and binarise relative to threshold
        flat = data.flatten()
        bind = {p: (np.pi if val > self.threshold else 0) for p, val in zip(self.input_params, flat)}
        # Evaluate expectation
        exp_val = self.estimator.run(self.circuit,
                                     parameter_binds=[bind],
                                     observables=[self.observable])[0]
        # Pass through linear head
        return float(self.linear(torch.tensor([exp_val], dtype=torch.float32)).item())

def Conv():
    """Factory returning a HybridQuantumFilter instance."""
    return HybridQuantumFilter()
