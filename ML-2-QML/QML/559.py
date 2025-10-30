"""QCNNHybridModel – a quantum convolutional neural network with a scikit‑learn‑style API.

The model builds a multi‑layer convolution‑pooling circuit, wraps it in an EstimatorQNN,
and exposes `fit`, `predict`, and `score` methods.  It also supports exporting
the trained ansatz to QASM for downstream use.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import QuantumKernelClassifier
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score
from typing import Iterable, Tuple


class QCNNHybridModel:
    """
    Quantum convolutional neural network implemented with Qiskit Machine Learning.

    Features
    --------
    * Configurable depth of convolutional and pooling layers.
    * Training via gradient descent on a quantum circuit (EstimatorQNN).
    * Optional use of a quantum‑kernel SVM.
    * Export of the trained circuit in QASM format.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        conv_depth: int = 3,
        pool_depth: int = 3,
        shots: int = 1024,
        backend=None,
        optimizer=None,
    ) -> None:
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.estimator = Estimator(backend=self.backend, shots=self.shots)
        self.optimizer = optimizer or COBYLA()
        self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        self.trained = False

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def _build_circuit(self) -> None:
        # Feature map
        self.feature_map = ZFeatureMap(self.num_qubits, reps=1, entanglement="full")
        # Ansatz (convolution + pooling)
        self.ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")
        for i in range(self.conv_depth):
            self.ansatz.compose(
                self._conv_layer(self.num_qubits // (2**i), f"c{i+1}"),
                list(range(self.num_qubits // (2**i))),
                inplace=True,
            )
            self.ansatz.compose(
                self._pool_layer(self.num_qubits // (2**i), f"p{i+1}"),
                list(range(self.num_qubits // (2**i))),
                inplace=True,
            )
        # Combine feature map and ansatz
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(self.num_qubits), inplace=True)
        # Observable (single‑qubit Z on first qubit)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q in range(0, num_qubits, 2):
            qc.compose(
                self._conv_circuit(params[3 * (q // 2) : (q // 2 + 3)]),
                [q, q + 1],
                inplace=True,
            )
        return qc

    def _conv_circuit(self, params) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        for idx, (src, dst) in enumerate(
            zip(range(0, num_qubits, 2), range(1, num_qubits, 2))
        ):
            qc.compose(
                self._pool_circuit(params[3 * idx : 3 * (idx + 1)]), [src, dst], inplace=True
            )
        return qc

    def _pool_circuit(self, params) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ------------------------------------------------------------------
    # Training and inference
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> "QCNNHybridModel":
        """
        Train the quantum neural network by optimizing the ansatz parameters.
        """
        self.qnn.fit(X, y, epochs=epochs, lr=lr, optimizer=self.optimizer, verbose=verbose)
        self.trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities of the positive class."""
        if not self.trained:
            raise RuntimeError("Model has not been trained yet.")
        probs = self.qnn.predict(X)
        return probs.reshape(-1, 1)

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict_classes(X)
        return accuracy_score(y, preds)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def export_qasm(self, path: str) -> None:
        """Export the trained ansatz circuit to a QASM file."""
        if not self.trained:
            raise RuntimeError("Model has not been trained yet.")
        with open(path, "w") as f:
            f.write(self.circuit.decompose().to_qasm())

def QCNN() -> QCNNHybridModel:
    """Factory returning a default QCNNHybridModel instance."""
    return QCNNHybridModel()
