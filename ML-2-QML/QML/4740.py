"""
Hybrid quantum estimator that constructs a QCNN‑style ansatz and augments it
with a self‑attention circuit.  The estimator is compatible with
``qiskit_machine_learning.neural_networks.EstimatorQNN`` and can be used
with any Qiskit backend.

Key components
--------------
* Feature map: ``ZFeatureMap`` on 8 qubits.
* Convolution + pooling layers: identical to the QCNN seed.
* Self‑attention layer: uses CRX gates to entangle adjacent qubits.
* Estimator: ``StatevectorEstimator`` from Qiskit.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


# -------------------------------------------------------------
# Convolution & pooling primitives (identical to QCNN)
# -------------------------------------------------------------
def conv_circuit(params: np.ndarray) -> QuantumCircuit:
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional block that applies `conv_circuit` to every adjacent pair."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc


def pool_circuit(params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling block that reduces the qubit count."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index: param_index + 3]), [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# -------------------------------------------------------------
# Self‑attention circuit (CRX entanglement) – inspired by SelfAttention QML
# -------------------------------------------------------------
def attention_circuit(params: np.ndarray) -> QuantumCircuit:
    """Self‑attention style circuit that rotates each qubit and adds CRX between neighbours."""
    qc = QuantumCircuit(8)
    for i in range(8):
        qc.rx(params[3 * i], i)
        qc.ry(params[3 * i + 1], i)
        qc.rz(params[3 * i + 2], i)
    # CRX between adjacent qubits
    for i in range(7):
        qc.crx(params[8 + i], i, i + 1)
    return qc


# -------------------------------------------------------------
# Full ansatz construction
# -------------------------------------------------------------
def build_ansatz() -> QuantumCircuit:
    """
    Builds a QCNN‑style ansatz with an additional attention sub‑circuit.
    """
    # Feature map + ansatz
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)

    # Second convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)

    # Third convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Attention layer appended at the end
    attention_params = ParameterVector("attn", length=8 * 3 + 7)
    ansatz.compose(attention_circuit(attention_params), range(8), inplace=True)

    return ansatz


# -------------------------------------------------------------
# Estimator construction
# -------------------------------------------------------------
def HybridEstimatorQNN() -> EstimatorQNN:
    """
    Creates a Qiskit EstimatorQNN that uses the hybrid ansatz.
    """
    # Backend and estimator
    backend = Aer.get_backend("statevector_simulator")
    estimator = StatevectorEstimator(backend=backend)

    # Build circuits and observable
    feature_map = ZFeatureMap(8)
    ansatz = build_ansatz()
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Combine feature map and ansatz
    full_circuit = QuantumCircuit(8)
    full_circuit.compose(feature_map, range(8), inplace=True)
    full_circuit.compose(ansatz, range(8), inplace=True)

    # Create EstimatorQNN
    qnn = EstimatorQNN(
        circuit=full_circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["HybridEstimatorQNN"]
