import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolution Layer")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[i*3:(i+2)*3])
        qc.append(sub, [i, i+1])
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(prefix, length=(num_qubits//2)*3)
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[(i//2)*3:((i//2)+1)*3])
        qc.append(sub, [i, i+1])
    return qc

def QCNNQuantum(num_qubits: int = 8) -> EstimatorQNN:
    estimator = StatevectorEstimator()
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer(num_qubits, "p1"), inplace=True)
    ansatz.compose(conv_layer(num_qubits//2, "c2"), inplace=True)
    ansatz.compose(pool_layer(num_qubits//2, "p2"), inplace=True)
    ansatz.compose(conv_layer(num_qubits//4, "c3"), inplace=True)
    ansatz.compose(pool_layer(num_qubits//4, "p3"), inplace=True)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

__all__ = ["QCNNQuantum"]
