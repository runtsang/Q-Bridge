"""Hybrid quantum neural network integrating QCNN ansatz with a self‑attention block."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def EstimatorQNN():
    algorithm_globals.random_seed = 12345  # deterministic behavior

    # Feature map
    feature_map = ZFeatureMap(8)

    # Parameter vectors
    conv_params = ParameterVector("c", length=8 * 3)   # 3 params per qubit for conv layers
    pool_params = ParameterVector("p", length=8 * 3)   # 3 params per qubit for pool layers
    sa_params   = ParameterVector("sa", length=8 * 3 + 7)  # 3 rotation per qubit + 7 entangle params

    # Helper to build a single‑qubit rotation block
    def rotate_block(i: int, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.rx(params[3 * i], 0)
        qc.ry(params[3 * i + 1], 0)
        qc.rz(params[3 * i + 2], 0)
        return qc

    # Convolution layer (3‑parameter rotation per qubit)
    def conv_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.compose(rotate_block(i, params), [i], inplace=True)
        return qc

    # Pooling layer (omit last Rz, identical to conv for simplicity)
    def pool_layer(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.compose(rotate_block(i, params), [i], inplace=True)
        return qc

    # Self‑attention block
    def attention_block(num_qubits: int, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.compose(rotate_block(i, params), [i], inplace=True)
        for i in range(num_qubits - 1):
            qc.crx(params[3 * num_qubits + i], i, i + 1)
        return qc

    # Build the ansatz
    ansatz = QuantumCircuit(8)

    # First convolution
    ansatz.compose(conv_layer(8, conv_params), list(range(8)), inplace=True)
    # First pooling
    ansatz.compose(pool_layer(8, pool_params), list(range(8)), inplace=True)

    # Second convolution (operate on 4 qubits)
    ansatz.compose(conv_layer(4, conv_params[24:]), list(range(4, 8)), inplace=True)
    # Second pooling
    ansatz.compose(pool_layer(4, pool_params[12:]), list(range(4, 8)), inplace=True)

    # Third convolution (operate on 2 qubits)
    ansatz.compose(conv_layer(2, conv_params[36:]), list(range(6, 8)), inplace=True)
    # Third pooling
    ansatz.compose(pool_layer(2, pool_params[18:]), list(range(6, 8)), inplace=True)

    # Append self‑attention
    ansatz.compose(attention_block(8, sa_params), list(range(8)), inplace=True)

    # Full circuit with feature map
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = Estimator()
    weight_params = conv_params + pool_params + sa_params

    qnn = QiskitEstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn
