"""Quantum estimator that mirrors the classical architecture defined in EstimatorQNNGen042.

The circuit encodes each feature into a qubit via RX rotations and builds a
depth‑controlled variational ansatz.  Observables are Pauli‑Z operators on each
qubit, providing a vector of expectation values that can be used for regression
or binary classification.  The class exposes the same metadata (encoding,
weight sizes, observables) as the classical version, enabling seamless hybrid
training pipelines.
"""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class EstimatorQNNGen042:
    """
    Quantum neural network that can be used as a regressor or a binary classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the dimensionality of the input data).
    depth : int, default 3
        Number of variational layers.  Each layer contains a single‑qubit RY
        rotation followed by a chain of CZ gates.
    task : str, {"regression", "classification"}, default "regression"
        The downstream objective.  For classification the expectation vector
        is interpreted as logits for a 2‑class softmax.
    """

    def __init__(self, num_qubits: int, depth: int = 3, task: str = "regression") -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be'regression' or 'classification'")
        self.task = task
        self.num_qubits = num_qubits
        self.depth = depth

        # Build the variational circuit
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        # Wrap into qiskit-machine-learning EstimatorQNN
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables – one Pauli‑Z per qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(weights), observables

    def get_estimator(self) -> EstimatorQNN:
        """Return the wrapped qiskit machine‑learning EstimatorQNN instance."""
        return self.estimator_qnn

    def get_encoding(self) -> List[ParameterVector]:
        """Return the list of feature parameters that drive the circuit."""
        return self.encoding

    def get_weights(self) -> List[ParameterVector]:
        """Return the list of variational parameters."""
        return self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of Pauli‑Z observables used for read‑out."""
        return self.observables

    def get_weight_sizes(self) -> List[int]:
        """Return the parameter count for each linear layer in the quantum circuit."""
        return [len(w) for w in self.weights]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, depth={self.depth}, task={self.task})"

__all__ = ["EstimatorQNNGen042"]
