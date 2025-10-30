import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
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
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[i*3:(i+1)*3])
        qc.compose(sub, [i, i+1], inplace=True)
        qc.barrier()
    return qc

def pool_layer(sources, sinks, prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk, idx in zip(sources, sinks, range(len(sources))):
        sub = pool_circuit(params[idx*3:(idx+1)*3])
        qc.compose(sub, [src, snk], inplace=True)
        qc.barrier()
    return qc

def self_attention_circuit(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Quantum self‑attention block using parameterized rotations and CNOT‑based entanglement."""
    qc = QuantumCircuit(num_qubits)
    rot_params = ParameterVector(prefix + "_rot", length=num_qubits * 3)
    ent_params = ParameterVector(prefix + "_ent", length=num_qubits - 1)
    for i in range(num_qubits):
        qc.rx(rot_params[3*i], i)
        qc.ry(rot_params[3*i + 1], i)
        qc.rz(rot_params[3*i + 2], i)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
        qc.rz(ent_params[i], i+1)  # simple entangling gate
    return qc

def QCNNQuantum() -> EstimatorQNN:
    """Builds a QCNN with an embedded self‑attention sub‑circuit."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)

    # Convolution + pooling stages
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Insert self‑attention block after the third pooling
    ansatz.compose(self_attention_circuit(8, "attn"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNQuantum"]
