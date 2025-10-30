import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

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

def conv_layer(num_qubits, prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i*3:(i+2)*3]), [i, i+1])
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for idx, (s, t) in enumerate(zip(sources, sinks)):
        qc.append(pool_circuit(params[idx*3:(idx+1)*3]), [s, t])
    return qc

def attention_layer(num_qubits, prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3 + num_qubits - 1)
    # Local rotations
    for i in range(num_qubits):
        qc.rx(params[i*3], i)
        qc.ry(params[i*3+1], i)
        qc.rz(params[i*3+2], i)
    # Entangling CRX gates
    for i in range(num_qubits - 1):
        qc.crx(params[num_qubits*3 + i], i, i+1)
    return qc

def QCNNQML() -> EstimatorQNN:
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    # First convolution + attention + pooling
    ansatz.append(conv_layer(8, "c1"), range(8))
    ansatz.append(attention_layer(8, "a1"), range(8))
    ansatz.append(pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8))
    # Second convolution + attention + pooling
    ansatz.append(conv_layer(4, "c2"), [4,5,6,7])
    ansatz.append(attention_layer(4, "a2"), [4,5,6,7])
    ansatz.append(pool_layer([4,5], [6,7], "p2"), [4,5,6,7])
    # Third convolution + attention + pooling
    ansatz.append(conv_layer(2, "c3"), [6,7])
    ansatz.append(attention_layer(2, "a3"), [6,7])
    ansatz.append(pool_layer([6], [7], "p3"), [6,7])
    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(ansatz, range(8))
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNQML"]
