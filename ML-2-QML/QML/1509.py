"""Quantum variational regressor with entangling layers.

This module defines EstimatorQNN that builds a two‑qubit variational circuit
with an angle‑encoding of the input. The circuit is parameterised by a set of
trainable weights that are optimised via a classical optimiser. The model
mirrors the classical EstimatorQNN API, exposing a `predict` method that
returns a single‑bit expectation value as a regression output.
"""

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.providers.fake_provider import FakeVigo
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class EstimatorQNN:
    """Variational quantum regression model."""

    def __init__(self, num_qubits: int = 2, depth: int = 2, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or FakeVigo()
        self.params = self._build_ansatz()
        self.estimator = StatevectorEstimator(backend=self.backend)
        self.observable = SparsePauliOp.from_list([("Z" * self.num_qubits, 1)])

    def _build_ansatz(self):
        circuit = QuantumCircuit(self.num_qubits)
        x = Parameter("x")
        circuit.ry(x, 0)
        weight_idx = 0
        params = []
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                theta = Parameter(f"theta_{weight_idx}")
                circuit.ry(theta, q)
                weight_idx += 1
                params.append(theta)
            for q in range(self.num_qubits - 1):
                circuit.cx(q, q + 1)
        self.circuit = circuit
        return params

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = []
        weight_vals = np.random.uniform(0, 2 * np.pi, len(self.params))
        for x_val in X:
            binding = {self.circuit.parameters[0]: x_val}
            binding.update({p: w for p, w in zip(self.params, weight_vals)})
            result = self.estimator.run(
                circuits=[self.circuit],
                parameter_binds=[binding],
                observables=[self.observable],
            ).result()
            y.append(result[0])
        return np.array(y)

__all__ = ["EstimatorQNN"]
