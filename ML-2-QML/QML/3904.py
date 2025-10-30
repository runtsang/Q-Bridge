"""
Hybrid QCNN built with Qiskit that embeds a classical‑style estimator circuit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QuantumEstimatorQNN

# --------------------------------------------------------------------------- #
#  Helper layers – convolution and pooling – adapted from the original QCNN
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block with 3 variational parameters."""
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
    """Builds a convolution block over an even number of qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    ptr = 0
    for i in range(0, num_qubits, 2):
        block = conv_circuit(params[ptr : ptr + 3])
        qc.append(block, [i, i + 1])
        qc.barrier()
        ptr += 3
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block – identical to conv but without the final Rz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Reduces the number of qubits by half using pooling blocks."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    ptr = 0
    for i in range(0, num_qubits, 2):
        block = pool_circuit(params[ptr : ptr + 3])
        qc.append(block, [i, i + 1])
        qc.barrier()
        ptr += 3
    return qc


# --------------------------------------------------------------------------- #
#  Hybrid ansatz – classical feature map + convolution / pooling layers
#  + a parameterised “EstimatorQNN” sub‑circuit acting as a read‑out layer.
# --------------------------------------------------------------------------- #
def hybrid_ansatz() -> QuantumCircuit:
    """Builds the full quantum ansatz for the QCNN."""
    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz with three convolution–pool cycles
    ansatz = QuantumCircuit(8, name="QCNN Ansatz")
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer(8, "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer(4, "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer(2, "p3"), inplace=True)

    # EstimatorQNN sub‑circuit – a single‑qubit parametric gate
    # replicated across all qubits to act as a read‑out layer
    est_params = ParameterVector("θ_est", length=8)
    est_circ = QuantumCircuit(8)
    for i, p in enumerate(est_params):
        est_circ.ry(p, i)
    ansatz.compose(est_circ, inplace=True)

    # Full circuit – feature map followed by ansatz
    full = QuantumCircuit(8)
    full.compose(feature_map, inplace=True)
    full.compose(ansatz, inplace=True)
    return full


# --------------------------------------------------------------------------- #
#  Build the EstimatorQNN object used as a hybrid quantum‑classical network
# --------------------------------------------------------------------------- #
def QCNN() -> QuantumEstimatorQNN:
    """Factory returning a hybrid EstimatorQNN with the QCNN ansatz."""
    # Build separate feature map and ansatz to split input/weight parameters
    feature_map = ZFeatureMap(8)
    ansatz = hybrid_ansatz()

    # Combine into a single circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    estimator = Estimator()

    qnn = QuantumEstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNN"]
