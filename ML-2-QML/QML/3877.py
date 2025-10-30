import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNN:
    """
    Quantum implementation of a QCNN that augments the classical ansatz
    with a parameterised quantum transformer layer.  The circuit
    is compatible with the EstimatorQNN interface and can be trained
    with any optimizer available in Qiskit Machine Learning.
    """

    def __init__(self, n_qubits: int = 8, feature_map_dim: int = 8) -> None:
        self.n_qubits = n_qubits
        self.feature_map_dim = feature_map_dim
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)]),
            input_params=ZFeatureMap(self.feature_map_dim).parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="ConvolutionalLayer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[param_index : param_index + 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="PoolingLayer")
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        param_index = 0
        for i in range(0, num_qubits, 2):
            sub = self._pool_circuit(params[param_index : param_index + 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            qc.barrier()
            param_index += 3
        return qc

    def _quantum_transformer_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="QuantumTransformerLayer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            qc.rx(params[i], i)
            qc.ry(params[num_qubits + i], i)
            qc.rz(params[2 * num_qubits + i], i)
        # Entangling pattern
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(num_qubits - 1, 0)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        # Feature map
        feature_map = ZFeatureMap(self.feature_map_dim)

        # Ansatz: convolution + pooling + transformer
        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")

        # First convolutional layer
        ansatz.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)

        # First pooling layer
        ansatz.compose(self._pool_layer(self.n_qubits, "p1"), inplace=True)

        # Second convolutional layer
        ansatz.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)

        # Second pooling layer
        ansatz.compose(self._pool_layer(self.n_qubits // 2, "p2"), inplace=True)

        # Third convolutional layer
        ansatz.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)

        # Third pooling layer
        ansatz.compose(self._pool_layer(self.n_qubits // 4, "p3"), inplace=True)

        # Quantum transformer layer
        ansatz.compose(self._quantum_transformer_layer(self.n_qubits, "qt"), inplace=True)

        # Final circuit
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit

    def predict(self, X):
        """Predict using the underlying EstimatorQNN."""
        return self.qnn.predict(X)

__all__ = ["HybridQCNN"]
