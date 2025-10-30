from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ------------------------------------------------------------
# Helper functions to build convolution and pooling layers
# ------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[3 * (i // 2) : 3 * (i // 2) + 3])
        qc.append(sub, [i, i+1])
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[3 * (i // 2) : 3 * (i // 2) + 3])
        qc.append(sub, [i, i+1])
    return qc

# ------------------------------------------------------------
# Quantum self‑attention block
# ------------------------------------------------------------
def quantum_self_attention(num_qubits: int) -> QuantumCircuit:
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    # Rotation parameters
    rot = ParameterVector("rot", length=3 * num_qubits)
    for i in range(num_qubits):
        qc.rx(rot[3 * i], i)
        qc.ry(rot[3 * i + 1], i)
        qc.rz(rot[3 * i + 2], i)
    # Entangling parameters
    ent = ParameterVector("ent", length=num_qubits - 1)
    for i in range(num_qubits - 1):
        qc.crx(ent[i], i, i + 1)
    qc.measure(qr, cr)
    return qc

# ------------------------------------------------------------
# Full QCNN + Attention circuit
# ------------------------------------------------------------
def QCNN() -> EstimatorQNN:
    # Feature map
    feature_map = ZFeatureMap(8)
    # Build ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    # First conv + pool
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), list(range(8)), inplace=True)
    # Second conv + pool
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), list(range(4, 8)), inplace=True)
    # Third conv + pool
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), list(range(6, 8)), inplace=True)
    # Insert quantum self‑attention after the ansatz
    attn = quantum_self_attention(4)
    ansatz.compose(attn, list(range(8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable for binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observable=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN"]
