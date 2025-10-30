"""Quantum QCNN hybrid model incorporating a random feature map and
convolutional ansatz inspired by QCNN and Quanvolution."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNNHybrid() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Random feature map: encode each qubit with a random rotation
    def random_feature_map(num_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            theta = ParameterVector(f'theta_{i}', 1)[0]
            qc.ry(theta, i)
            qc.rz(theta, i)
        # random entanglement
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        return qc

    feature_map = random_feature_map(8)

    # Convolutional layer as before
    def conv_layer(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer_block(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_layer(param_vec[idx:idx+3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    # Pooling layer
    def pool_layer(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer_block(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = pool_layer(param_vec[idx:idx+3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    # Build ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.append(conv_layer_block(8, "c1"), range(8))
    ansatz.append(pool_layer_block(8, "p1"), range(8))
    ansatz.append(conv_layer_block(4, "c2"), range(4, 8))
    ansatz.append(pool_layer_block(4, "p2"), range(4, 8))
    ansatz.append(conv_layer_block(2, "c3"), range(6, 8))
    ansatz.append(pool_layer_block(2, "p3"), range(6, 8))

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(ansatz, range(8))

    # Observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
