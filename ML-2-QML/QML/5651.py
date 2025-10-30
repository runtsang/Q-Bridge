"""Hybrid QCNN quantum circuit with enhanced variational layers and multi‑observable output."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit_v2(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit variational circuit with an extra RZ gate on qubit 1.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    # Extra parameterised gate
    qc.rz(params[3], 1)
    return qc

def conv_layer_v2(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """
    Convolutional layer built from the enhanced two‑qubit circuit.
    """
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    # each pair needs 4 parameters now
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 4)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit_v2(params[param_index:param_index+4]), [q1, q2])
        qc.barrier()
        param_index += 4
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit_v2(params[param_index:param_index+4]), [q1, q2])
        qc.barrier()
        param_index += 4
    return qc

def pool_circuit_v2(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling circuit with an additional RZ gate on qubit 1.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    # extra gate
    qc.rz(params[3], 1)
    return qc

def pool_layer_v2(sources, sinks, param_prefix: str) -> QuantumCircuit:
    """
    Pooling layer that pairs source and sink qubits using the enhanced pooling circuit.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 4)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit_v2(params[param_index:param_index+4]), [source, sink])
        qc.barrier()
        param_index += 4
    return qc

def QCNNHybrid(measure_x: bool = False) -> EstimatorQNN:
    """
    Build a hybrid QCNN ansatz with the enhanced variational layers and
    return an EstimatorQNN.  When ``measure_x`` is True the observable
    is a sum of X operators instead of Z.
    """
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Hybrid Ansatz")

    # First convolution and pooling
    ansatz.compose(conv_layer_v2(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer_v2([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)

    # Second convolution and pooling
    ansatz.compose(conv_layer_v2(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer_v2([0,1], [2,3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolution and pooling
    ansatz.compose(conv_layer_v2(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer_v2([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: sum of single‑qubit Z or X depending on measure_x
    pauli_char = "X" if measure_x else "Z"
    labels = [(pauli_char + "I"*7, 1)]
    for i in range(1,8):
        labels.append(("I"*i + pauli_char + "I"*(7-i), 1))
    observable = SparsePauliOp.from_list(labels)

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNHybrid"]
