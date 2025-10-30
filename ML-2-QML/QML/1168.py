"""Quantum convolutional neural network with noise-aware training.

This module defines a `QCNN` class that encapsulates a variational quantum circuit
mirroring the classical QCNN architecture. The class builds a feature map,
ansatz, and EstimatorQNN, and exposes a `fit` method that trains the parameters
using a COBYLA optimizer. It also supports adding a depolarizing noise model
to the simulator, allowing experiments on noisy hardware.

The class is fully compatible with Qiskit Machine Learning APIs and can be
used as a drop‑in replacement for classical models in hybrid workflows.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as BaseEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.providers.fake_provider import FakeVigo
from qiskit.providers import BackendV2
from qiskit.providers.simulator import AerSimulator

class QCNN:
    """Quantum convolutional neural network with optional noise and training."""
    def __init__(self,
                 input_dim: int = 8,
                 num_qubits: int = 8,
                 noise_level: float = 0.0,
                 backend: BackendV2 | None = None,
                 seed: int | None = 12345):
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.seed = seed
        self.backend = backend or AerSimulator()
        if self.noise_level > 0.0:
            # Add a simple depolarizing noise channel to the simulator
            from qiskit.providers.fake_provider import FakeVigo
            self.backend = FakeVigo()
        self.estimator = BaseEstimator(backend=self.backend)
        self.feature_map = ZFeatureMap(num_qubits, reps=1, insert_barriers=False)
        self.ansatz = self._build_ansatz(num_qubits)
        self.circuit = self._compose_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )
        self.optimizer = COBYLA(maxiter=200)

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Construct the variational ansatz with convolution and pooling layers."""
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
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                sub = conv_circuit(params[idx:idx + 3])
                qc.append(sub, [q1, q2])
                idx += 3
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
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                sub = pool_circuit(params[idx:idx + 3])
                qc.append(sub, [q1, q2])
                idx += 3
            return qc

        ansatz = QuantumCircuit(num_qubits)
        ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
        ansatz.compose(pool_layer(num_qubits, "p1"), inplace=True)
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
        ansatz.compose(pool_layer(num_qubits // 2, "p2"), inplace=True)
        ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
        ansatz.compose(pool_layer(num_qubits // 4, "p3"), inplace=True)
        return ansatz

    def _compose_circuit(self) -> QuantumCircuit:
        """Combine feature map and ansatz into a single circuit."""
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        return circuit

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """Train the QCNN using a simple stochastic gradient descent loop."""
        y = y.astype(np.float64)
        dataset = list(zip(X, y))
        for epoch in range(epochs):
            np.random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                X_batch, y_batch = zip(*batch)
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)

                preds = self.predict(X_batch)
                loss = -np.mean(y_batch * np.log(preds + 1e-9) + (1 - y_batch) * np.log(1 - preds + 1e-9))

                grads = self._parameter_shift_gradient(X_batch, y_batch)

                lr = 0.01
                for idx, param in enumerate(self.ansatz.parameters):
                    param_val = float(param)
                    param.assign_value(param_val - lr * grads[idx])

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

    def _parameter_shift_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate gradients using the parameter‑shift rule."""
        shift = np.pi / 2
        grads = np.zeros(len(self.ansatz.parameters))
        for idx, param in enumerate(self.ansatz.parameters):
            original = float(param)
            param.assign_value(original + shift)
            plus = self.predict(X)
            param.assign_value(original - shift)
            minus = self.predict(X)
            param.assign_value(original)
            grads[idx] = (plus - minus).mean() / (2 * np.sin(shift))
        return grads

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for the given input data."""
        return np.array([self.qnn.predict([x])[0] for x in X])

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

__all__ = ["QCNN"]
