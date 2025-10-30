"""
QCNNHybridQML: Quantum circuit that mirrors the classical QCNN architecture
and returns a variational expectation value.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# 1. Quantum convolution block
# --------------------------------------------------------------------------- #
def _conv_block(params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(len(qubits))
    idx = 0
    for q in qubits:
        qc.rz(-np.pi / 2, q)
        qc.cx(q, qubits[(idx + 1) % len(qubits)])
        qc.rz(params[idx], q)
        qc.ry(params[idx + 1], qubits[(idx + 1) % len(qubits)])
        qc.cx(qubits[(idx + 1) % len(qubits)], q)
        qc.ry(params[idx + 2], qubits[(idx + 1) % len(qubits)])
        qc.cx(q, qubits[(idx + 1) % len(qubits)])
        qc.rz(np.pi / 2, q)
        idx += 3
    return qc

# --------------------------------------------------------------------------- #
# 2. Quantum pooling block
# --------------------------------------------------------------------------- #
def _pool_block(params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(len(qubits))
    idx = 0
    for q in qubits:
        qc.rz(-np.pi / 2, q)
        qc.cx(q, qubits[(idx + 1) % len(qubits)])
        qc.rz(params[idx], q)
        qc.ry(params[idx + 1], qubits[(idx + 1) % len(qubits)])
        qc.cx(qubits[(idx + 1) % len(qubits)], q)
        qc.ry(params[idx + 2], qubits[(idx + 1) % len(qubits)])
        idx += 3
    return qc

# --------------------------------------------------------------------------- #
# 3. Build full QCNN‑style ansatz
# --------------------------------------------------------------------------- #
def _build_qcnn_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    # Layer 1
    params = ParameterVector("c1", length=num_qubits * 3)
    qc.append(_conv_block(params, list(range(num_qubits))), range(num_qubits))
    # Pool 1
    params = ParameterVector("p1", length=(num_qubits // 2) * 3)
    qc.append(_pool_block(params, list(range(num_qubits))), range(num_qubits))
    # Layer 2
    params = ParameterVector("c2", length=(num_qubits // 2) * 3)
    qc.append(_conv_block(params, list(range(num_qubits // 2, num_qubits))), range(num_qubits))
    # Pool 2
    params = ParameterVector("p2", length=(num_qubits // 4) * 3)
    qc.append(_pool_block(params, list(range(num_qubits // 2, num_qubits))), range(num_qubits))
    # Layer 3
    params = ParameterVector("c3", length=(num_qubits // 4) * 3)
    qc.append(_conv_block(params, list(range(num_qubits // 4, num_qubits))), range(num_qubits))
    # Pool 3
    params = ParameterVector("p3", length=(num_qubits // 8) * 3)
    qc.append(_pool_block(params, list(range(num_qubits // 4, num_qubits))), range(num_qubits))
    return qc

# --------------------------------------------------------------------------- #
# 4. Full QCNN‑style QNN
# --------------------------------------------------------------------------- #
def QCNNHybridQML(num_qubits: int = 8) -> EstimatorQNN:
    """
    Factory that returns a variational quantum neural network
    mirroring the QCNN architecture.  It combines a Z‑feature map
    with a convolution‑pooling ansatz and a single‑qubit observable.
    """
    feature_map = ZFeatureMap(num_qubits)
    ansatz = _build_qcnn_ansatz(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNHybridQML"]
