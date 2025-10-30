import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import algorithm_globals

def QCNNEnhanced() -> EstimatorQNN:
    """
    Quantum convolutional neural network with a 12‑qubit hierarchical architecture.
    The circuit consists of three convolutional layers followed by pooling, mirroring
    a classical CNN but using parameterized two‑qubit blocks. The feature map is a
    ZFeatureMap over 12 qubits.
    """
    algorithm_globals.random_seed = 42
    estimator = Estimator()

    # Two‑qubit convolutional block
    def conv_block(params: ParameterVector) -> QuantumCircuit:
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

    # Convolutional layer over a list of qubit pairs
    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            block = conv_block(params[i // 2 * 3 : i // 2 * 3 + 3])
            qc.append(block, [i, i + 1])
        return qc

    # Two‑qubit pooling block
    def pool_block(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Pooling layer over a list of qubit pairs
    def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            block = pool_block(params[i // 2 * 3 : i // 2 * 3 + 3])
            qc.append(block, [i, i + 1])
        return qc

    # Feature map over 12 qubits
    feature_map = ZFeatureMap(num_qubits=12, reps=1, entanglement='linear')
    feature_map = feature_map.decompose()

    # Ansatz construction
    ansatz = QuantumCircuit(12, name="QCNN_Ansatz")
    ansatz.compose(conv_layer(12, "c1"), range(12), inplace=True)
    ansatz.compose(pool_layer(12, "p1"), range(12), inplace=True)
    ansatz.compose(conv_layer(6, "c2"), range(6), inplace=True)
    ansatz.compose(pool_layer(6, "p2"), range(6), inplace=True)
    ansatz.compose(conv_layer(3, "c3"), range(3), inplace=True)
    ansatz.compose(pool_layer(3, "p3"), range(3), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)

    # Observable for binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * 11, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn
