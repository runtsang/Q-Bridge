"""Variational quantum regressor with entangling layers and custom observable.

The model mirrors the original EstimatorQNN but now uses a parameterised
quantum circuit.  It exposes the same EstimatorQNN class name so that
experiments can swap the backend without changing client code.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN:
    """
    Variational quantum circuit for regression.

    Parameters
    ----------
    qubits : int
        Number of qubits in the ansatz.
    layers : int
        Number of variational layers.
    """

    def __init__(self, qubits: int = 1, layers: int = 2) -> None:
        self.qubits = qubits
        self.layers = layers

        # Build parameterised circuit
        self.circuit = self._build_circuit()
        # Define observable (Pauli-Y on all qubits)
        self.observable = SparsePauliOp.from_list(
            [(("Y" * self.qubits), 1)]
        )
        # Wrap with Qiskit EstimatorQNN
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a layered ansatz with RY/RY gates and CNOT entanglement."""
        qc = QuantumCircuit(self.qubits)
        param_iter = iter(Parameter("w{}").format(i) for i in range(self.qubits * self.layers + self.qubits))
        # Input layer (single parameter per qubit)
        for q in range(self.qubits):
            qc.ry(next(param_iter), q)
        # Variational layers
        for _ in range(self.layers):
            # Rotation layer
            for q in range(self.qubits):
                qc.ry(next(param_iter), q)
            # Entanglement (chain)
            for q in range(self.qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def get_estimator(self):
        """Return the underlying Qiskit EstimatorQNN instance."""
        return self.estimator_qnn

__all__ = ["EstimatorQNN"]
