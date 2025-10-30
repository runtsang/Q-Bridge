"""Quantum implementation of a QCNN with a self‑attention layer."""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used by the QCNN ansatz."""
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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolution layer that composes `_conv_circuit` over all qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i:i+3])
        qc.append(sub, [i, i+1])
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    """Pooling layer that composes `_pool_circuit` over source‑sink pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[:3])
        qc.append(sub, [src, snk])
        params = params[3:]
    return qc

def _attention_circuit(params: ParameterVector) -> QuantumCircuit:
    """Attention subcircuit applying rotations and CRX entanglement."""
    num_qubits = params.length // (3 + 1)  # 3 rotations + 1 entangle per qubit
    qc = QuantumCircuit(num_qubits)
    # Rotations
    for i in range(num_qubits):
        qc.rx(params[3*i], i)
        qc.ry(params[3*i+1], i)
        qc.rz(params[3*i+2], i)
    # Entanglement (CRX between consecutive qubits)
    for i in range(num_qubits - 1):
        qc.crx(params[3*num_qubits + i], i, i+1)
    return qc

def _attention_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Wraps the attention circuit into an instruction."""
    qc = QuantumCircuit(num_qubits, name="Attention Layer")
    params = ParameterVector(param_prefix, length=3*num_qubits + (num_qubits - 1))
    sub = _attention_circuit(params)
    qc.append(sub.to_instruction(), range(num_qubits))
    return qc

def QCNNGen320() -> EstimatorQNN:
    """
    Builds a QCNN ansatz that includes a self‑attention layer.
    Returns an EstimatorQNN ready for training with a classical optimizer.
    """
    estimator = Estimator()

    # Feature map for classical data
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Insert attention after the final pooling
    ansatz.compose(_attention_layer(8, "attn1"), range(8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: measure the first qubit in the Z basis
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn

__all__ = ["QCNNGen320", "QCNNGen320"]
