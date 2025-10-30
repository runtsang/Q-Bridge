from __future__ import annotations

import numpy as np
import torch
from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
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

def _conv_layer(num_qubits, name_prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(name_prefix, length=(num_qubits//2)*3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(num_qubits, name_prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(name_prefix, length=(num_qubits//2)*3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _pool_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _build_qcnn_circuit() -> QuantumCircuit:
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit.decompose()

def HybridQCNN() -> EstimatorQNN:
    """Quantum QCNN implemented as a variational QNN."""
    estimator = Estimator()
    circuit = _build_qcnn_circuit()
    observable = SparsePauliOp.from_list([("Z"*8, 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=ZFeatureMap(8).parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

def qnn_kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Compute kernel Gram matrix using the QCNN ansatz."""
    qnn = HybridQCNN()
    return np.array([[qnn(torch.tensor(x.reshape(1, -1), dtype=torch.float32),
                         torch.tensor(y.reshape(1, -1), dtype=torch.float32)
                        ).item() for y in b] for x in a])

__all__ = ["HybridQCNN", "qnn_kernel_matrix"]
