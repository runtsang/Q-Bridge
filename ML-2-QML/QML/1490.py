import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params):
    """Two‑qubit convolution block with trainable rotations and entangling gates."""
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
    """Two‑qubit pooling block, removing one qubit from the feature map."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits, param_prefix):
    """Stack of convolution blocks applied pairwise across the qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolution Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc


def pool_layer(sources, sinks, param_prefix):
    """Pooling across selected qubit pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc


class QCNNHybrid:
    """Hybrid QCNN combining a parameterized ansatz with a classical post‑processing layer."""
    def __init__(self):
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Ansatz construction
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        # Full circuit
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, range(8), inplace=True)
        self.circuit.compose(ansatz, range(8), inplace=True)
        self.circuit.decompose()
        # Observable for classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        # Estimator
        self.estimator = Estimator()
        # QNN wrapper
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator
        )
        # Classical post‑processing weights
        self.classical_weight = np.random.randn()
        self.classical_bias = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions for input X (shape: n_samples × 8)."""
        exp_vals = self.qnn.predict(X)
        logits = exp_vals * self.classical_weight + self.classical_bias
        probs = 1 / (1 + np.exp(-logits))
        return probs

    def __repr__(self):
        return f"QCNNHybrid(qnn={self.qnn})"


def QCNN() -> QCNNHybrid:
    """Factory returning the configured QCNNHybrid."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNN"]
