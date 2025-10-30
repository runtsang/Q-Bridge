from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN__gen229:
    """
    Quantum estimator that mirrors the classical EstimatorQNN__gen229.
    Supports regression (Y measurement) and classification (Z measurement).
    """

    def __init__(self, num_qubits: int = 2, depth: int = 1, task: str = "regression") -> None:
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits (features).
        depth: int
            Number of variational layers.
        task: str
            "regression" or "classification".
        """
        self.task = task
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit(
            num_qubits, depth, task
        )
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=estimator,
        )

    def _build_circuit(
        self, num_qubits: int, depth: int, task: str
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """
        Build a layered ansatz with data‑encoding and variational parameters.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        # Data encoding
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Y for regression, Z for classification
        if task == "regression":
            observables = [SparsePauliOp.from_list([("Y" * num_qubits, 1)])]
        else:
            observables = [SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)
                                                    for i in range(num_qubits)])]

        return qc, list(encoding), list(weights), observables

    def predict(self, data: list[float]) -> float | list[float]:
        """
        Evaluate the quantum estimator on a single data point.
        """
        return self.estimator_qnn.predict(data)[0]

    def get_encoding(self) -> list[int]:
        return list(self.encoding)

    def get_weight_sizes(self) -> list[int]:
        return [len(w) for w in self.weights]

    def get_observables(self) -> list[SparsePauliOp]:
        return self.observables

__all__ = ["EstimatorQNN__gen229"]
