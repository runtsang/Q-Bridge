"""
Quantum hybrid QCNN implementation.

The module defines a `HybridQCNN` class that constructs a variational
circuit comprising:
  * a Z‑feature map,
  * three convolutional layers,
  * three pooling layers,
  * and an optional quantum fully‑connected layer.
The class exposes `run` for inference and `train` for variational
training using a COBYLA optimizer via Qiskit Machine Learning.
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
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from typing import Iterable, Sequence, Optional

# --------------------------------------------------------------------------- #
# Quantum surrogate for the fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFullyConnectedLayer:
    """
    Simple parameterised quantum circuit that emulates a fully‑connected layer.
    """
    def __init__(self, n_qubits: int, backend: qiskit.providers.BaseBackend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

# --------------------------------------------------------------------------- #
# Hybrid QCNN quantum circuit
# --------------------------------------------------------------------------- #
class HybridQCNN:
    """
    Quantum hybrid QCNN that integrates convolution, pooling, a feature
    map, and an optional quantum fully‑connected layer.
    """
    def __init__(self, n_qubits: int = 8,
                 backend: Optional[qiskit.providers.BaseBackend] = None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.feature_map = ZFeatureMap(n_qubits)
        self._build_ansatz()
        self._build_qnn()

    # --------------------------------------------------------------------- #
    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            conv = self._conv_circuit(params[param_index:param_index + 3])
            qc.append(conv, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            conv = self._conv_circuit(params[param_index:param_index + 3])
            qc.append(conv, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _conv_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int],
                    param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix,
                                 length=(num_qubits // 2) * 3)
        for src, sink in zip(sources, sinks):
            pool = self._pool_circuit(params[param_index:param_index + 3])
            qc.append(pool, [src, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _build_ansatz(self) -> None:
        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
                       list(range(8)), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"),
                       list(range(4, 8)), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"),
                       list(range(6, 8)), inplace=True)
        self.ansatz = ansatz

    def _build_qnn(self) -> None:
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, range(self.n_qubits), inplace=True)
        circuit.compose(self.ansatz, range(self.n_qubits), inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator()
        )

    # --------------------------------------------------------------------- #
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward evaluation of the quantum hybrid QCNN.

        Parameters
        ----------
        inputs : np.ndarray
            Feature array of shape (num_samples, n_qubits).

        Returns
        -------
        np.ndarray
            Predicted expectation values from the quantum circuit.
        """
        return self.qnn.predict(inputs)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50) -> NeuralNetworkClassifier:
        """
        Train the QCNN using a COBYLA optimizer.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Labels.
        epochs : int
            Number of optimization iterations.

        Returns
        -------
        NeuralNetworkClassifier
            Trained model.
        """
        optimizer = COBYLA(maxiter=epochs)
        clf = NeuralNetworkClassifier(optimizer=optimizer, estimator=self.qnn)
        clf.fit(X, y)
        return clf
