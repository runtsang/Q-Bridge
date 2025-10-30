"""Enhanced QCNN-inspired variational quantum circuit with adaptive pooling."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import algorithm_globals
import numpy as np

def QCNNplus() -> EstimatorQNN:
    """Build an enhanced QCNN-inspired variational quantum circuit."""
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    def _conv_unit(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            idx = i // 2
            qc.append(_conv_unit(param_vec[idx * 3: idx * 3 + 3]), [i, i + 1])
        return qc

    def _pool_unit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            idx = i // 2
            qc.append(_pool_unit(param_vec[idx * 3: idx * 3 + 3]), [i, i + 1])
        return qc

    # Build a three‑layer hierarchical ansatz
    ansatz = QuantumCircuit(8)

    # Layer 1: convolution over 4 qubits
    ansatz.compose(conv_layer(4, "c1"), list(range(4)), inplace=True)
    # Layer 1: pooling over 2 qubits
    ansatz.compose(pool_layer(2, "p1"), [0, 1], inplace=True)

    # Layer 2: convolution over 2 qubits
    ansatz.compose(conv_layer(2, "c2"), [2, 3], inplace=True)
    # Layer 2: pooling over 1 qubit
    ansatz.compose(pool_layer(1, "p2"), [4], inplace=True)

    # Layer 3: convolution over 2 qubits
    ansatz.compose(conv_layer(2, "c3"), [6, 7], inplace=True)
    # Layer 3: pooling over 1 qubit
    ansatz.compose(pool_layer(1, "p3"), [6], inplace=True)

    # Feature map
    feature_map = ZFeatureMap(8)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNplus", "EstimatorQNN"]
