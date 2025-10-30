"""Quantum convolutional neural network using Pennylane and Qiskit backend."""

import pennylane as qml
import pennylane.numpy as np
from pennylane import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA


def _conv_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN ansatz."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    idx = 0
    for q in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub.to_instruction(), [q, q + 1])
        idx += 3
    return qc


def _pool_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Pooling block that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    idx = 0
    for sink, source in zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits)):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[idx], 0)
        sub.ry(params[idx + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[idx + 2], 1)
        qc.append(sub.to_instruction(), [source, sink])
        idx += 3
    return qc


def QCNN() -> EstimatorQNN:
    """Return a variational QCNN implemented with Pennylane and Qiskit."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)
    num_qubits = 8

    # Build ansatz circuit
    ansatz = QuantumCircuit(num_qubits)
    # Layer 1: conv (8 qubits)
    ansatz.compose(_conv_block(num_qubits, "c1"), range(num_qubits), inplace=True)
    # Layer 2: pool (8 qubits -> 4)
    ansatz.compose(_pool_block(num_qubits, "p1"), range(num_qubits), inplace=True)
    # Layer 3: conv (4 qubits)
    ansatz.compose(_conv_block(4, "c2"), range(4, 8), inplace=True)
    # Layer 4: pool (4 qubits -> 2)
    ansatz.compose(_pool_block(4, "p2"), range(4, 8), inplace=True)
    # Layer 5: conv (2 qubits)
    ansatz.compose(_conv_block(2, "c3"), range(6, 8), inplace=True)
    # Layer 6: pool (2 qubits -> 1)
    ansatz.compose(_pool_block(2, "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
