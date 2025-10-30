"""Quantum QCNN implementation using Qiskit EstimatorQNN with parameter‑shift."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _pool_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def QCNNQML() -> EstimatorQNN:
    """Constructs a hybrid QCNN ansatz with a compact feature map and parameter‑shift gradients."""
    estimator = Estimator(shots=1024)
    feature_map = ZFeatureMap(8, reps=1, paulis="ZZ")
    ansatz = QuantumCircuit(8)
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), range(8), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), range(4, 8), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), range(6, 8), inplace=True)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        gradient_method="parameter-shift",
    )
    return qnn

__all__ = ["QCNNQML", "QCNNQML"]
