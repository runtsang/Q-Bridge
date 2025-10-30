import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_block(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the ansatz."""
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

def _pool_block(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Apply a convolution block to every adjacent pair of qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_block(params[i // 2 * 3 : (i // 2 + 1) * 3])
        qc.append(block, [i, i + 1])
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Apply a pooling block to every adjacent pair of qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        block = _pool_block(params[i // 2 * 3 : (i // 2 + 1) * 3])
        qc.append(block, [i, i + 1])
    return qc

def build_qcnn_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Construct a layered ansatz that alternates convolution and pooling
    operations, mirroring the classical QCNN structure.
    """
    qc = QuantumCircuit(num_qubits)
    # Feature map
    fm = ZFeatureMap(num_qubits)
    qc.append(fm, range(num_qubits))

    # Alternating conv / pool layers
    for d in range(depth):
        qc.append(conv_layer(num_qubits, f"c{d}"), range(num_qubits))
        qc.append(pool_layer(num_qubits, f"p{d}"), range(num_qubits))
    return qc

def create_qcnn_qnn(num_qubits: int,
                    depth: int,
                    shots: int = 1024) -> EstimatorQNN:
    """
    Wrap the QCNN ansatz in an EstimatorQNN for use with PyTorch optimizers.
    """
    estimator = Estimator()
    circuit = build_qcnn_ansatz(num_qubits, depth)
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=ZFeatureMap(num_qubits).parameters,
        weight_params=[p for p in circuit.parameters if "c" in p or "p" in p],
        estimator=estimator,
    )
    return qnn

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Build a simple layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, encoding parameters, weight parameters, and observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

__all__ = [
    "create_qcnn_qnn",
    "build_qcnn_ansatz",
    "build_classifier_circuit",
]
