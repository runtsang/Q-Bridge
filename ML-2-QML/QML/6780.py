import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

def QCNNSA() -> EstimatorQNN:
    """Quantum QCNN augmented with a parameterised self‑attention subcircuit."""
    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolution block
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        return qc

    def conv_layer(num_qubits, name):
        qc = QuantumCircuit(num_qubits, name=name)
        params = ParameterVector(name, length=(num_qubits//2) * 3)
        idx = 0
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[idx:idx+3])
            qc.append(sub, [i, i+1])
            qc.barrier()
            idx += 3
        return qc

    # Pooling block
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, name):
        num_qubits = len(sources)+len(sinks)
        qc = QuantumCircuit(num_qubits, name=name)
        params = ParameterVector(name, length=len(sources)*3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = pool_circuit(params[idx:idx+3])
            qc.append(sub, [s, t])
            qc.barrier()
            idx += 3
        return qc

    # Self‑attention subcircuit
    def self_attention_layer(num_qubits, name):
        qc = QuantumCircuit(num_qubits, name=name)
        params = ParameterVector(name, length=num_qubits*3)
        for i in range(num_qubits):
            qc.rx(params[3*i], i)
            qc.ry(params[3*i+1], i)
            qc.rz(params[3*i+2], i)
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
        return qc

    # Assemble ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4,8)), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6,8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)
    # Add self‑attention
    ansatz.compose(self_attention_layer(8, "sa"), list(range(8)), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz).decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn
