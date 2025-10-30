"""Quantum QCNN model with extended ansatz and feature map.

This module builds a hybrid quantum neural network that mirrors the classical
architecture but replaces each linear block with a parameterised quantum
circuit. The ansatz consists of multiple convolutional and pooling layers
defined over 8 qubits. The feature map uses a RealAmplitudes layer with
entanglement and two repetitions for richer data embedding. The model
exposes a simple predict interface and can be trained with Qiskit's
COBYLA optimizer.

Typical usage
-------------
>>> model = QCNN()
>>> probs = model.predict(X)   # X is a NumPy array of shape (n_samples, 8)
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as PrimitiveEstimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator

class QCNNModel:
    """Quantum QCNN model wrapping an EstimatorQNN.

    Parameters
    ----------
    backend : str, default "qasm_simulator"
        Backend name for the Aer simulator.
    shots : int, default 1024
        Number of shots for the simulator.
    device : str, default "cpu"
        Device for the quantum instance (ignored for Aer).
    """

    def __init__(
        self,
        backend: str = "qasm_simulator",
        shots: int = 1024,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.shots = shots
        self.device = device

        # Ensure reproducibility
        algorithm_globals.random_seed = 12345

        # Primitive estimator
        self.estimator = PrimitiveEstimator(
            backend=AerSimulator(),  # use Aer for fast simulation
            shots=shots,
        )

        # Build the QNN
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> EstimatorQNN:
        """Construct the quantum neural network."""
        # Feature map: RealAmplitudes with entanglement and two repetitions
        feature_map = RealAmplitudes(8, entanglement="linear", reps=2)

        # Convolutional unitary (two‑qubit block)
        def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

        # Convolutional layer over many qubits
        def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for idx, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
                sub = conv_circuit(params[idx * 3 : idx * 3 + 3])
                qc.append(sub, [q1, q2])
                qc.barrier()
            return qc

        # Pooling unitary (two‑qubit measurement‑preserving block)
        def pool_circuit(params: ParameterVector) -> QuantumCircuit:
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        # Pooling layer
        def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for idx, (q1, q2) in enumerate(zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits))):
                sub = pool_circuit(params[idx * 3 : idx * 3 + 3])
                qc.append(sub, [q1, q2])
                qc.barrier()
            return qc

        # Build ansatz with three convolutional & pooling stages
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer(8, "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer(4, "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer(2, "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        # Observable for binary classification
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Decompose to avoid data copying overhead
        circuit = circuit.decompose()

        return EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for input data X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 8)
            Feature matrix.
        """
        preds = self.qnn.predict(X)
        return preds.reshape(-1)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the QNN using COBYLA to minimise binary cross‑entropy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 8)
            Training features.
        y : np.ndarray, shape (n_samples,)
            Binary labels.
        """
        optimizer = COBYLA(maxiter=200)
        classifier = NeuralNetworkClassifier(optimizer=optimizer, estimator_qnn=self.qnn)
        # Note: the classifier expects a QuantumInstance; we provide a dummy one
        qi = QuantumInstance(AerSimulator(), shots=self.shots, device=self.device)
        classifier.fit(X, y, quantum_instance=qi)

def QCNN() -> QCNNModel:
    """Factory returning an instance of :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
