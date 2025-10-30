"""Hybrid quantum estimator combining a parameterised circuit with a self‑attention block.

The quantum side mirrors the classical self‑attention pattern and uses Qiskit
to evaluate expectation values over a parameterised circuit, enabling hybrid
training with the classical feed‑forward network.
"""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class HybridQuantumEstimator:
    """Hybrid quantum estimator combining a parameterised circuit with a self‑attention block."""
    def __init__(self, input_dim=2, n_qubits=4):
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        # Parameters
        self.input_params = [Parameter(f"x{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits)]
        self.attn_params = [Parameter(f"a{i}") for i in range(n_qubits*(n_qubits-1))]

        # Build base circuit
        self.circuit = QuantumCircuit(n_qubits)
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i % n_qubits)  # simple encoding

        # Self‑attention entanglement
        for i in range(n_qubits-1):
            self.circuit.crx(self.attn_params[i], i, i+1)

        # Add weight rotations
        for i, p in enumerate(self.weight_params):
            self.circuit.rz(p, i)

        # Observable
        self.observable = SparsePauliOp.from_list([("Y"*n_qubits, 1)])

        # Estimator
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, input_vals, weight_vals, attn_vals, shots=1024):
        """Execute the hybrid circuit and return expectation value."""
        param_bindings = {
            **{p: v for p, v in zip(self.input_params, input_vals)},
            **{p: v for p, v in zip(self.weight_params, weight_vals)},
            **{p: v for p, v in zip(self.attn_params, attn_vals)},
        }
        result = self.estimator_qnn.evaluate([param_bindings], shots=shots)
        return result[0]  # expectation value

def EstimatorQNN():
    """Compatibility wrapper returning the hybrid quantum estimator."""
    return HybridQuantumEstimator()

__all__ = ["HybridQuantumEstimator", "EstimatorQNN"]
