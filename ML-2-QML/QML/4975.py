"""Quantum hybrid QCNN.

This implementation builds a Qiskit EstimatorQNN that mirrors
the classical HybridQCNN structure:
* Feature map: ZFeatureMap (8 qubits)
* QCNN‑style convolution & pooling layers
* A fraud‑like parametric sub‑circuit
* A quantum‑like fully‑connected block (parameterised RX/RY/CX)
* Classical self‑attention applied to the measurement vector
"""

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning import utils as qml_utils
from qiskit import Aer

# --------------------------------------------------------------------------- #
# Helper: QCNN convolution and pooling circuits (adapted from QCNN.py)
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        start = i // 2 * 3
        qc.append(_conv_circuit(params[start:start+3]), [i, i+1])
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        start = i // 2 * 3
        qc.append(_pool_circuit(params[start:start+3]), [i, i+1])
    return qc

# --------------------------------------------------------------------------- #
# Fraud‑like parametric sub‑circuit (simplified)
# --------------------------------------------------------------------------- #
def fraud_layer(params: ParameterVector, wires: Sequence[int]) -> QuantumCircuit:
    """A 2‑qubit fraud‑style gate that mimics BS + RZ + RY."""
    qc = QuantumCircuit(len(wires))
    for i, w in enumerate(wires):
        qc.rx(params[3*i], w)
        qc.ry(params[3*i + 1], w)
        qc.rz(params[3*i + 2], w)
    return qc

# --------------------------------------------------------------------------- #
# Quantum‑like fully‑connected block
# --------------------------------------------------------------------------- #
def quantum_fc_block(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=3 * num_qubits)
    for i in range(num_qubits):
        qc.rx(params[3*i], i)
        qc.ry(params[3*i + 1], i)
        qc.rz(params[3*i + 2], i)
    # entangling layer
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

# --------------------------------------------------------------------------- #
# Self‑attention post‑processing (classical)
# --------------------------------------------------------------------------- #
def classical_self_attention(q_values: np.ndarray, fraud_values: np.ndarray) -> np.ndarray:
    """Simple dot‑product attention between QCNN outputs and fraud outputs."""
    scores = np.exp(q_values @ fraud_values.T)  # (B, B)
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores @ fraud_values  # (B, B)

# --------------------------------------------------------------------------- #
# Full hybrid QNN
# --------------------------------------------------------------------------- #
def HybridQCNNQ() -> EstimatorQNN:
    """Return a Qiskit EstimatorQNN that implements the hybrid QCNN."""
    # Feature map
    feature_map = ZFeatureMap(8)
    # Build ansatz
    circuit = QuantumCircuit(8)
    # 1st conv & pool
    circuit.append(conv_layer(8, "c1"), range(8))
    circuit.append(pool_layer(8, "p1"), range(8))
    # 2nd conv & pool
    circuit.append(conv_layer(4, "c2"), range(4,8))
    circuit.append(pool_layer(4, "p2"), range(4,8))
    # 3rd conv & pool
    circuit.append(conv_layer(2, "c3"), range(6,8))
    circuit.append(pool_layer(2, "p3"), range(6,8))
    # Fraud layer (2 qubits)
    circuit.append(fraud_layer(ParameterVector("f1", length=6), [0,1]), [0,1])
    # Quantum‑like FC block
    circuit.append(quantum_fc_block(4, "qfc"), [0,1,2,3])
    # Measurement observable (Z on qubit 0)
    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    # Wrap in EstimatorQNN
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQCNNQ"]
