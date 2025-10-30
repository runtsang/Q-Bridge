"""Quantum QCNN generator using Qiskit’s EstimatorQNN.

The architecture is a direct quantum analogue of the classical QCNNGen208.
It constructs a feature map, a depth‑controlled convolution‑pooling ansatz,
and wraps it in an EstimatorQNN for training with classical optimisers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals


class QCNNGen208:
    """Quantum QCNN generator.

    Parameters
    ----------
    num_qubits : int, default 8
        Number of qubits representing the feature vector.
    depth : int, default 3
        Number of convolution–pooling pairs in the ansatz.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3) -> None:
        algorithm_globals.random_seed = 12345
        self.num_qubits = num_qubits
        self.depth = depth
        self.estimator = Estimator()
        self.circuit, self.encoding, self.weights, self.observables = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=self.estimator,
        )

    def _conv_block(self, qc: QuantumCircuit, params: ParameterVector, qubits: list[int]) -> None:
        """Append a two‑qubit convolution block to *qc*."""
        for i in range(0, len(qubits), 2):
            rc = QuantumCircuit(2, name="conv")
            rc.rz(-np.pi / 2, 1)
            rc.cx(1, 0)
            rc.rz(params[3 * i], 0)
            rc.ry(params[3 * i + 1], 1)
            rc.cx(0, 1)
            rc.ry(params[3 * i + 2], 1)
            rc.cx(1, 0)
            rc.rz(np.pi / 2, 0)
            qc.append(rc.to_instruction(), [qubits[i], qubits[i + 1]])
            qc.barrier()

    def _pool_block(self, qc: QuantumCircuit, params: ParameterVector, qubits: list[int]) -> None:
        """Append a two‑qubit pooling block to *qc*."""
        for i in range(0, len(qubits), 2):
            rc = QuantumCircuit(2, name="pool")
            rc.rz(-np.pi / 2, 1)
            rc.cx(1, 0)
            rc.rz(params[3 * i], 0)
            rc.ry(params[3 * i + 1], 1)
            rc.cx(0, 1)
            rc.ry(params[3 * i + 2], 1)
            qc.append(rc.to_instruction(), [qubits[i], qubits[i + 1]])
            qc.barrier()

    def _build_ansatz(self) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
        """Construct the feature‑map + ansatz circuit."""
        # Feature map – identical to the classical 8‑dimensional input
        feature_map = ZFeatureMap(self.num_qubits)
        # Ansatz
        ansatz = QuantumCircuit(self.num_qubits, name="ansatz")
        # Encoding
        encoding = ParameterVector("x", self.num_qubits)
        for q, p in zip(range(self.num_qubits), encoding):
            ansatz.rx(p, q)
        # Variational parameters
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        # Build depth‑controlled conv‑pool layers
        idx = 0
        for _ in range(self.depth):
            # Convolution stage
            self._conv_block(ansatz, weights[idx: idx + self.num_qubits], list(range(self.num_qubits)))
            idx += self.num_qubits
            # Pooling stage – keep half the qubits
            self._pool_block(ansatz, weights[idx: idx + self.num_qubits], list(range(self.num_qubits)))
            idx += self.num_qubits
        # Observables: single‑qubit Z on each qubit
        observables = [
            SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - i - 1), 1)])
            for i in range(self.num_qubits)
        ]
        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        return circuit, list(encoding), list(weights), observables

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run the EstimatorQNN on *inputs* and return raw expectation values."""
        return self.qnn.predict(inputs)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
        """Train the QNN using COBYLA."""
        opt = COBYLA(maxiter=epochs * 10)
        nn = NeuralNetworkClassifier(
            estimator=self.estimator,
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            optimizer=opt,
            loss="cross_entropy",
            num_classes=2,
        )
        nn.fit(X, y)


__all__ = ["QCNNGen208"]
