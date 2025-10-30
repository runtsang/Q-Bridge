from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap

__all__ = ["build_classifier_circuit"]

def _conv_block(qubits: List[int], params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(*qubits)
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(np.pi / 2, qubits[0])
    return qc

def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    n_pairs = num_qubits // 2
    params = ParameterVector(prefix, length=n_pairs * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub_params = params[idx:idx+3]
        qc.compose(_conv_block([i, i+1], sub_params), inplace=True)
        idx += 3
    return qc

def _pool_block(qubits: List[int], params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(*qubits)
    qc.rz(-np.pi / 2, qubits[1])
    qc.cx(qubits[1], qubits[0])
    qc.rz(params[0], qubits[0])
    qc.ry(params[1], qubits[1])
    qc.cx(qubits[0], qubits[1])
    qc.ry(params[2], qubits[1])
    return qc

def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    n_pairs = num_qubits // 2
    params = ParameterVector(prefix, length=n_pairs * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub_params = params[idx:idx+3]
        qc.compose(_pool_block([i, i+1], sub_params), inplace=True)
        idx += 3
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a QCNN‑style ansatz with the given number of qubits and depth.
    Returns:
        - circuit: the full quantum circuit (feature map + ansatz)
        - encoding: ParameterVector for data encoding
        - weight_params: ParameterVector for variational parameters
        - observables: list of SparsePauliOp to be measured
    """
    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Ansatz
    ansatz = QuantumCircuit(num_qubits)
    for d in range(depth):
        ansatz.compose(conv_layer(num_qubits, f"c{d}"), inplace=True)
        ansatz.compose(pool_layer(num_qubits, f"p{d}"), inplace=True)

    # Combine
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observables: measure Z on each qubit
    observables = [
        SparsePauliOp.from_list([("Z" + "I" * (num_qubits - i - 1), 1)])
        for i in range(num_qubits)
    ]

    encoding = feature_map.parameters
    weight_params = ansatz.parameters
    return circuit, encoding, weight_params, observables
