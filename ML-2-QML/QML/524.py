"""Quantum CNN model with scikit‑learn‑style API.

The class builds on the original QCNN code but adds a lightweight training
loop and parameter management.  It exposes ``fit`` and ``predict`` methods
compatible with scikit‑learn estimators, enabling it to be used directly in
pipelines or cross‑validation.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.algorithms.optimizers import COBYLA, SPSA
from typing import Any, Dict


class QCNNHybrid:
    """
    Quantum CNN with scikit‑learn‑style API.

    Parameters
    ----------
    input_dim : int
        Number of input features (must match the feature map).
    optimizer : str, default="COBYLA"
        Optimiser to use for training.  Supported values: "COBYLA", "SPSA".
    maxiter : int, default=200
        Maximum number of optimizer iterations.
    """

    def __init__(
        self,
        input_dim: int = 8,
        optimizer: str = "COBYLA",
        maxiter: int = 200,
    ) -> None:
        self.input_dim = input_dim
        self.optimizer_name = optimizer.upper()
        self.maxiter = maxiter
        self._build_circuit()
        self._build_qnn()
        self._optimizer = self._select_optimizer()

    def _build_circuit(self) -> None:
        """Builds the QCNN ansatz as in the seed."""
        # Feature map
        self.feature_map = ZFeatureMap(num_qubits=self.input_dim)

        # Convolution and pooling sub‑circuits
        def conv_circuit(params):
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(params[i * 3 : (i + 2) * 3])
                qc.append(sub, [i, i + 1])
                qc.barrier()
            return qc

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
            for i in range(0, num_qubits, 2):
                sub = pool_circuit(params[(i // 2) * 3 : ((i // 2) + 1) * 3])
                qc.append(sub, [i, i + 1])
                qc.barrier()
            return qc

        # Assemble ansatz
        ansatz = QuantumCircuit(self.input_dim, name="Ansatz")
        ansatz.compose(conv_layer(self.input_dim, "c1"), inplace=True)
        ansatz.compose(pool_layer(self.input_dim, "p1"), inplace=True)
        ansatz.compose(conv_layer(self.input_dim // 2, "c2"), inplace=True)
        ansatz.compose(pool_layer(self.input_dim // 2, "p2"), inplace=True)
        ansatz.compose(conv_layer(self.input_dim // 4, "c3"), inplace=True)
        ansatz.compose(pool_layer(self.input_dim // 4, "p3"), inplace=True)

        self.ansatz = ansatz

        # Observable
        self.observable = SparsePauliOp.from_list(
            [("Z" + "I" * (self.input_dim - 1), 1)]
        )

    def _build_qnn(self) -> None:
        """Instantiate the EstimatorQNN."""
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )
        # Flatten weight parameters for optimizer
        self.n_params = len(self.ansatz.parameters)
        self.params = np.zeros(self.n_params)

    def _select_optimizer(self):
        """Return an optimizer instance."""
        if self.optimizer_name == "COBYLA":
            return COBYLA(maxiter=self.maxiter)
        if self.optimizer_name == "SPSA":
            return SPSA(maxiter=self.maxiter)
        raise ValueError(f"Unsupported optimizer {self.optimizer_name}")

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error loss on the training data."""
        self.qnn.set_weight_params(params)
        preds = self.qnn.predict(X)
        return np.mean((preds - y) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QCNNHybrid":
        """Train the QCNN with the chosen optimiser."""
        # Initial parameters
        self.params = np.random.randn(self.n_params) * 0.1
        # Wrap in a lambda for the optimizer
        def objective(p):
            return self._loss(p, X, y)

        result = self._optimizer.minimize(objective, self.params)
        self.params = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""
        self.qnn.set_weight_params(self.params)
        return self.qnn.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {"params": self.params}

    def set_params(self, **kwargs) -> None:
        if "params" in kwargs:
            self.params = kwargs["params"]

def QCNN() -> QCNNHybrid:
    """Factory returning a QCNNHybrid instance with default settings."""
    return QCNNHybrid()


__all__ = ["QCNN", "QCNNHybrid"]
