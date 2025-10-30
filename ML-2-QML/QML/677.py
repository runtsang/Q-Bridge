"""
Advanced quantum estimator based on a 3‑qubit entangled variational circuit.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNAdvanced:
    """
    Quantum neural network that extends the original single‑qubit example.
    Features:
        * 3 qubits with a user‑configurable number of entangling layers.
        * Ry, Rz, Rx rotations per qubit per layer.
        * Cyclic CNOT entanglement between qubits.
        * Observable on each qubit (Pauli‑Y) to provide multiple expectation values.
    """

    def __init__(self, input_dim: int = 2, n_qubits: int = 3, n_layers: int = 2) -> None:
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def _build_circuit(self) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        # Parameters for input encoding and variational weights
        input_params = ParameterVector("input", self.input_dim)
        weight_params = ParameterVector("weight",
                                        self.n_qubits * self.n_layers * 3)  # 3 params per qubit per layer
        qc = QuantumCircuit(self.n_qubits)

        # Encode inputs with Ry rotations
        for i in range(self.input_dim):
            qc.ry(input_params[i], i)

        # Variational layers with entanglement
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                idx = 3 * (layer * self.n_qubits + q)
                qc.ry(weight_params[idx], q)
                qc.rz(weight_params[idx + 1], q)
                qc.rx(weight_params[idx + 2], q)
            # Entangle qubits in a ring
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.n_qubits - 1, 0)

        return qc, input_params, weight_params

    def _build_observables(self) -> list[SparsePauliOp]:
        # Pauli‑Y observable on each qubit
        observables = []
        for q in range(self.n_qubits):
            pauli_str = ["I"] * self.n_qubits
            pauli_str[q] = "Y"
            observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1)]))
        return observables

    def get_estimator_qnn(self) -> EstimatorQNN:
        qc, input_params, weight_params = self._build_circuit()
        observables = self._build_observables()
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

def EstimatorQNNAdvanced() -> EstimatorQNN:
    """
    Factory returning an instance of the advanced quantum estimator.
    """
    model = EstimatorQNNAdvanced()
    return model.get_estimator_qnn()

__all__ = ["EstimatorQNNAdvanced"]
