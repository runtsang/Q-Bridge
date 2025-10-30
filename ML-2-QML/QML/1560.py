"""Quantum QCNN implemented with EstimatorQNN and a stochastic‑gradient training loop."""
from __future__ import annotations

import numpy as np
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA, GradientDescent
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_aer import AerSimulator

class QCNNModel:
    """
    Wrapper around a Qiskit EstimatorQNN that mimics the classical QCNN architecture.

    Features
    --------
    * A parameterised ansatz built from convolution and pooling blocks.
    * A custom training loop that supports either COBYLA or stochastic‑gradient descent.
    * A `predict` API returning state‑vector probabilities for binary classification.

    The model is intentionally lightweight: the ansatz uses 8 qubits and a 3‑parameter convolution block,
    matching the depth of the original seed while offering a clear separation of feature mapping,
    convolution, and pooling stages.
    """

    def __init__(self,
                 *, seed: int | None = None,
                 backend_name: str = "aer_simulator",
                 max_iter: int = 200,
                 optimizer: str = "cobyla",
                 learning_rate: float = 0.01) -> None:
        """
        Parameters
        ----------
        seed
            Random seed for reproducibility. If ``None``, no seed is set.
        backend_name
            Qiskit backend identifier (e.g. ``"aer_simulator"``).
        max_iter
            Maximum iterations for the optimiser.
        optimizer
            ``"cobyla"`` or ``"sgd"``.
        learning_rate
            Step size for SGD.
        """
        self.seed = seed
        if self.seed is not None:
            algorithm_globals.random_seed = self.seed
        self.backend = AerSimulator(method="statevector", seed_simulator=self.seed)
        self.estimator = Estimator(backend=self.backend)

        # Feature map and observables
        self.feature_map = ZFeatureMap(8)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Build ansatz
        self.ansatz = self._build_ansatz()

        # Assemble EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

        # Optimiser selection
        self.max_iter = max_iter
        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == "cobyla":
            self.optimizer = COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == "sgd":
            self.optimizer = GradientDescent(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    # --------------------------------------------------------------------------- #
    #  Convolution & pooling primitives
    # --------------------------------------------------------------------------- #
    def _conv_circuit(self, params: Sequence[Parameter]) -> QuantumCircuit:
        """Three‑parameter 2‑qubit convolution block."""
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

    def _pool_circuit(self, params: Sequence[Parameter]) -> QuantumCircuit:
        """Three‑parameter 2‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Construct a convolution layer over `num_qubits` qubits."""
        qc = QuantumCircuit(num_qubits, name="conv")
        qubits = list(range(num_qubits))
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[idx: idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
        """Construct a pooling layer mapping `sources` to `sinks`."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="pool")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, sink, p in zip(sources, sinks, params):
            sub = self._pool_circuit([p])
            qc.append(sub, [src, sink])
            qc.barrier()
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Create the full ansatz comprising three conv‑pool pairs."""
        qc = QuantumCircuit(8)

        # First conv‑pool
        qc.append(self._conv_layer(8, "c1"), range(8))
        qc.append(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8))

        # Second conv‑pool
        qc.append(self._conv_layer(4, "c2"), range(4, 8))
        qc.append(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8))

        # Third conv‑pool
        qc.append(self._conv_layer(2, "c3"), range(6, 8))
        qc.append(self._pool_layer([0], [1], "p3"), range(6, 8))

        return qc

    # --------------------------------------------------------------------------- #
    #  Prediction & training
    # --------------------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the model’s binary probability for each sample in ``X``.

        Parameters
        ----------
        X
            Input feature array of shape ``(n_samples, 8)``.
        """
        return self.qnn.predict(X)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the QNN using the selected optimiser.

        Parameters
        ----------
        X
            Training features, shape ``(n_samples, 8)``.
        y
            Binary labels, shape ``(n_samples,)``.
        """
        if self.optimizer_name == "cobyla":
            self.optimizer.minimize(
                self.qnn,
                initial_point=[0.0] * len(self.qnn.weight_params),
                objective_fn=lambda x: -np.mean(self.qnn.predict(X) * y + (1 - self.qnn.predict(X)) * (1 - y)),
            )
        else:  # SGD
            for _ in range(self.max_iter):
                gradients = self.optimizer.gradient(
                    self.qnn,
                    lambda x: -np.mean(self.qnn.predict(X) * y + (1 - self.qnn.predict(X)) * (1 - y)),
                )
                self.qnn.weight_params = [w + g for w, g in zip(self.qnn.weight_params, gradients)]

def QCNN() -> QCNNModel:
    """
    Factory returning a fully configured :class:`QCNNModel`.

    The returned object exposes ``predict`` and ``train`` methods and is ready
    for immediate use on 8‑dimensional binary classification tasks.
    """
    return QCNNModel(seed=12345, optimizer="cobyla", max_iter=200)

__all__ = ["QCNNModel", "QCNN"]
