import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit.primitives import Estimator as EstimatorPrimitive

# ---------------------------------------------------------------------------
# Helper functions that mirror the seed QCNN implementation
# ---------------------------------------------------------------------------

def conv_circuit(params):
    """Two‑qubit convolution block used in all layers."""
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
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def self_attention_layer(num_qubits, param_prefix):
    """
    Implements a lightweight self‑attention style sub‑circuit.
    Each qubit receives a rotation and adjacent qubits are entangled
    via a controlled‑RX gate.
    """
    qc = QuantumCircuit(num_qubits, name="SelfAttention Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3 + (num_qubits - 1))
    idx = 0
    # Rotations
    for i in range(num_qubits):
        qc.rx(params[idx], i); idx += 1
        qc.ry(params[idx], i); idx += 1
        qc.rz(params[idx], i); idx += 1
    # Entanglement
    for i in range(num_qubits - 1):
        qc.crx(params[idx], i, i + 1); idx += 1
    return qc

def conv_layer(num_qubits, param_prefix):
    """Wraps conv_circuit into a layer that acts on all even‑odd pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.compose(conv_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
        idx += 3
    return qc

def pool_layer(num_qubits, param_prefix):
    """Wraps pool_circuit into a layer that acts on all even‑odd pairs."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.compose(pool_circuit(params[idx:idx+3]), [q1, q2], inplace=True)
        idx += 3
    return qc

# ---------------------------------------------------------------------------
# Quantum QCNN with self‑attention and hybrid head
# ---------------------------------------------------------------------------

def QCNNGen525QNN() -> EstimatorQNN:
    """
    Constructs a quantum neural network that emulates the classical
    QCNNGen525Model.  The ansatz consists of alternating
    convolution, self‑attention, and pooling layers, followed by a
    simple read‑out observable.  The network is wrapped in an
    EstimatorQNN so that it can be trained end‑to‑end with gradient‑based
    optimizers.
    """
    # Feature map – encode input data into qubit states
    feature_map = ZFeatureMap(8, reps=1, entanglement='full')
    # Build the ansatz
    ansatz = QuantumCircuit(8)
    # First convolution + self‑attention + pooling
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(self_attention_layer(8, "sa1"), range(8), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), range(8), inplace=True)
    # Second convolution + self‑attention + pooling
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(self_attention_layer(4, "sa2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), range(4, 8), inplace=True)
    # Third convolution + self‑attention + pooling
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(self_attention_layer(2, "sa3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable – simple Z on qubit 0
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator used by EstimatorQNN
    estimator = EstimatorPrimitive()

    # Wrap everything into EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def QCNNGen525() -> EstimatorQNN:
    """
    Public factory that returns the quantum neural network.
    Mirrors the API of the classical factory.
    """
    return QCNNGen525QNN()


__all__ = ["QCNNGen525QNN", "QCNNGen525"]
