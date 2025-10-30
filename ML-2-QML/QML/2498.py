import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        sub = pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def quanvolution_feature_map(n_wires, param_prefix):
    qc = QuantumCircuit(n_wires)
    params = ParameterVector(param_prefix, length=n_wires)
    for i in range(n_wires):
        qc.ry(params[i], i)
    return qc

def QCNN() -> EstimatorQNN:
    estimator = Estimator()
    n_qubits = 8
    feature_map = quanvolution_feature_map(n_qubits, "x")
    ansatz = QuantumCircuit(n_qubits)
    ansatz.compose(conv_layer(n_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)
    circuit = QuantumCircuit(n_qubits)
    circuit.compose(feature_map, range(n_qubits), inplace=True)
    circuit.compose(ansatz, range(n_qubits), inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits-1), 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

class QuanvolutionQCNNHybrid:
    def __init__(self):
        self.qnn = QCNN()

    def get_qnn(self):
        return self.qnn

__all__ = ["QuanvolutionQCNNHybrid"]
