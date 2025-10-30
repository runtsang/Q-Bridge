import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals

class QCNN:
    """
    Quantum Convolutional Neural Network (QCNN) implementation.
    Builds a variational circuit with convolution and pooling layers,
    wraps it into an EstimatorQNN, and provides training and inference
    utilities. The architecture mirrors the original seed but adds
    a lightweight training loop using a classical optimizer.
    """

    def __init__(self,
                 num_qubits: int = 8,
                 seed: int = 12345,
                 backend: str ='statevector',
                 optimizer_name: str = 'COBYLA',
                 max_iter: int = 200,
                 learning_rate: float = 0.01) -> None:
        algorithm_globals.random_seed = seed
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
        self.optimizer = self._get_optimizer(optimizer_name, max_iter, learning_rate)

    def _get_optimizer(self, name: str, max_iter: int, lr: float):
        if name.upper() == 'COBYLA':
            return COBYLA(maxiter=max_iter)
        else:
            raise ValueError(f"Unsupported optimizer {name}")

    def _conv_circuit(self, params):
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

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[param_index:param_index+3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = self._conv_circuit(params[param_index:param_index+3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index:param_index+3])
            qc.append(sub, [src, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _build_circuit(self):
        self.feature_map = ZFeatureMap(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)

        # First Pooling Layer
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)

        # Second Pooling Layer
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)

        # Third Convolutional Layer
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)

        # Third Pooling Layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        self.ansatz = ansatz
        return circuit

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """
        Simple training loop using the specified optimizer.
        X shape: (n_samples, num_qubits)
        y shape: (n_samples,)
        """
        X_flat = X.reshape(-1, self.num_qubits)
        y_flat = y.reshape(-1)
        clf = NeuralNetworkClassifier(
            estimator=self.qnn,
            optimizer=self.optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        clf.fit(X_flat, y_flat)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class."""
        X_flat = X.reshape(-1, self.num_qubits)
        probs = self.qnn.predict(X_flat)
        return probs

    def get_parameters(self) -> np.ndarray:
        """Return current weight parameters."""
        return self.qnn.get_weights()

    def set_parameters(self, params: np.ndarray):
        """Set weight parameters."""
        self.qnn.set_weights(params)

def create_qcnn() -> QCNN:
    """Factory returning a default‑configured QCNN instance."""
    return QCNN()

__all__ = ["QCNN", "create_qcnn"]
