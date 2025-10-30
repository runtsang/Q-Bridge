import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params):
    """Two‑qubit unitary used in convolution layers."""
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

def pool_circuit(params):
    """Two‑qubit unitary used in pooling layers."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, shared_params: ParameterVector) -> QuantumCircuit:
    """Convolutional layer that applies the same set of parameters to each pair."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub_params = shared_params[param_index:param_index + 3]
        sub_circ = conv_circuit(sub_params)
        qc.append(sub_circ.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_layer(num_qubits: int, shared_params: ParameterVector) -> QuantumCircuit:
    """Pooling layer that applies the same set of parameters to each pair."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub_params = shared_params[param_index:param_index + 3]
        sub_circ = pool_circuit(sub_params)
        qc.append(sub_circ.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

class QCNNEnhanced:
    """Quantum convolutional neural network with parameter‑sharing and
    adaptive pooling.

    The architecture mirrors the classical counterpart but uses a
    parameter‑shared convolutional ansatz and a pooling stage that
    discards qubits while preserving learned correlations.
    """
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.estimator = Estimator()

        # Shared parameters for all convolution and pooling layers
        self.conv_params = ParameterVector("c", length=num_qubits * 3)
        self.pool_params = ParameterVector("p", length=(num_qubits // 2) * 3)

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)

        # Build ansatz
        self.ansatz = self._build_ansatz()

        # Observable (sum of Z on all qubits)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)])

        # QNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.conv_params + self.pool_params,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Feature map
        qc.compose(self.feature_map, range(self.num_qubits), inplace=True)

        # Convolution + pooling stages
        qc.compose(conv_layer(self.num_qubits, self.conv_params), range(self.num_qubits), inplace=True)
        qc.compose(pool_layer(self.num_qubits, self.pool_params), range(self.num_qubits), inplace=True)

        # After first pooling we effectively work on half the qubits
        half = self.num_qubits // 2
        qc.compose(conv_layer(half, self.conv_params), range(half), inplace=True)
        qc.compose(pool_layer(half, self.pool_params), range(half), inplace=True)

        # After second pooling we work on a quarter of the qubits
        quarter = half // 2
        qc.compose(conv_layer(quarter, self.conv_params), range(quarter), inplace=True)
        qc.compose(pool_layer(quarter, self.pool_params), range(quarter), inplace=True)

        return qc

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        return self.qnn.predict(inputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Alias to __call__ for clarity."""
        return self.__call__(inputs)

def QCNN() -> QCNNEnhanced:
    """Factory returning the configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()

__all__ = ["QCNN", "QCNNEnhanced"]
