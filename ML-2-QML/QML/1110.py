"""Quantum QCNN implementation using Qiskit EstimatorQNN with noise and gradient support."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator

class QCNNQuantum(EstimatorQNN):
    """
    A wrapper around :class:`qiskit_machine_learning.neural_networks.EstimatorQNN`
    that exposes a higher‑level API for training, noise injection and
    parameter‑shift gradient evaluation.

    Parameters
    ----------
    feature_map : QuantumCircuit
        Feature encoding circuit.
    ansatz : QuantumCircuit
        Variational ansatz implementing convolution and pooling layers.
    simulator : AerSimulator, optional
        Backend used for state‑vector estimation. Defaults to a noiseless
        simulator; a noisy model can be passed to enable realistic training.
    """

    def __init__(
        self,
        feature_map: QuantumCircuit,
        ansatz: QuantumCircuit,
        simulator: AerSimulator | None = None,
    ) -> None:
        if simulator is None:
            simulator = AerSimulator(method="statevector")
        estimator = Estimator(quantum_instance=QuantumInstance(simulator))
        observable = SparsePauliOp.from_list([("Z" + "I" * (feature_map.num_qubits - 1), 1)])
        super().__init__(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.simulator = simulator

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> None:
        """
        Simple stochastic gradient descent training loop using
        the parameter‑shift rule implemented by the Estimator.

        Parameters
        ----------
        X : ndarray
            Input data of shape (n_samples, n_features).
        y : ndarray
            Binary labels of shape (n_samples,).
        epochs : int
            Number of full passes over the training data.
        lr : float
            Learning rate for the weight updates.
        """
        weight_vals = np.random.randn(len(self.weight_params))
        for epoch in range(epochs):
            loss = 0.0
            for x, target in zip(X, y):
                pred = self.predict(x, weight_vals)
                loss += (pred - target) ** 2
                grad = self.gradient(x, weight_vals)
                weight_vals -= lr * grad
            loss /= len(X)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:02d} – MSE: {loss:.4f}")

    def predict(self, x: np.ndarray, weight_vals: np.ndarray) -> float:
        """Return the expectation value for a single sample."""
        return float(
            self.predict_quantum(
                input_values=dict(zip(self.input_params, x)),
                weight_values=dict(zip(self.weight_params, weight_vals)),
            )
        )

    def gradient(
        self,
        x: np.ndarray,
        weight_vals: np.ndarray,
    ) -> np.ndarray:
        """Compute the parameter‑shift gradient for a single sample."""
        return np.array(
            self.gradient_quantum(
                input_values=dict(zip(self.input_params, x)),
                weight_values=dict(zip(self.weight_params, weight_vals)),
            )
        )

def QCNN() -> QCNNQuantum:
    """
    Factory that builds a QCNNQuantum instance with the canonical
    feature‑map, convolution and pooling layers.
    """
    # Feature map – 8‑qubit Z‑feature map
    feature_map = ZFeatureMap(8)

    # Convolution & pooling blocks
    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = _conv_circuit(params[i * 3 : (i + 2) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        params = ParameterVector(prefix, length=num // 2 * 3)
        for src, sink in zip(sources, sinks):
            sub = _pool_circuit(params[:3])
            qc.append(sub, [src, sink])
            params = params[3:]
        return qc

    def _conv_circuit(params: list[ParameterVector]) -> QuantumCircuit:
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

    def _pool_circuit(params: list[ParameterVector]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Assemble ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    return QCNNQuantum(feature_map, ansatz)
