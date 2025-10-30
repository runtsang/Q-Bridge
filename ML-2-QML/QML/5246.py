"""Quantum circuit for QCNNGen130 – convolution, pooling, and a transformer‑like entangling block."""

import numpy as np
import qiskit as qk
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNN():
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Convolution on a pair of qubits
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

    # Convolution layer across qubit pairs
    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(conv_circuit(params[idx:idx + 3]), [q1, q2])
            idx += 3
        return qc

    # Pooling on a pair of qubits
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Pooling layer across qubit pairs
    def pool_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for src, dst in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(pool_circuit(params[idx:idx + 3]), [src, dst])
            idx += 3
        return qc

    # Transformer‑like entangling block
    def transformer_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 2)
        for i in range(num_qubits):
            qc.ry(params[2 * i], i)
            qc.rz(params[2 * i + 1], i)
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        qc.cx(num_qubits - 1, 0)
        return qc

    # Construct the full ansatz
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)

    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)

    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)

    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    ansatz.compose(transformer_layer(8, "t1"), inplace=True)

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
