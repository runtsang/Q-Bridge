"""Quantum helper for the Unified QCNN‑Kernel‑Classifier."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit convolution unit from the original QCNN paper.
    """
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

def conv_layer(num_qubits: int, depth: int, param_prefix: str) -> QuantumCircuit:
    """
    Builds a convolutional layer that applies ``conv_circuit`` to each adjacent
    pair of qubits, repeating the pattern for ``depth`` times.
    """
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        params = ParameterVector(f"{param_prefix}_{d}", length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_circuit(params[idx:idx+3])
            qc.compose(sub, [q1, q2], inplace=True)
            idx += 3
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """
    Two‑qubit pooling unit that discards one qubit via measurement.
    """
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: List[int], sinks: List[int], depth: int, param_prefix: str) -> QuantumCircuit:
    """
    Applies pooling to pairs defined by ``sources`` and ``sinks``.
    """
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    for d in range(depth):
        params = ParameterVector(f"{param_prefix}_{d}", length=len(sources) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = pool_circuit(params[idx:idx+3])
            qc.compose(sub, [src, sink], inplace=True)
            idx += 3
    return qc

def build_qcnn_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Assemble the full QCNN ansatz with alternating convolution and pooling layers.
    """
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits)

    # Convolution + pooling stages
    ansatz.compose(conv_layer(num_qubits, depth, "c1"), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), depth, "p1"), inplace=True)

    # Reduce qubit count for next stage
    remaining = num_qubits // 2
    if remaining > 1:
        ansatz.compose(conv_layer(remaining, depth, "c2"), inplace=True)
        ansatz.compose(pool_layer(list(range(remaining // 2)), list(range(remaining // 2, remaining)), depth, "p2"), inplace=True)

    # Final layer
    ansatz.compose(conv_layer(remaining // 2, depth, "c3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit.decompose()

def build_qcnn_qnn(num_qubits: int, depth: int, estimator: Estimator | None = None) -> EstimatorQNN:
    """
    Wrap the QCNN ansatz into an EstimatorQNN ready for training.
    """
    if estimator is None:
        estimator = Estimator()
    circuit = build_qcnn_ansatz(num_qubits, depth)
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=ZFeatureMap(num_qubits).parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

def quantum_kernel(num_qubits: int) -> QuantumCircuit:
    """
    Simple quantum kernel circuit used to approximate an RBF‑style kernel.
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(0.1, i)  # small rotation for feature encoding
    return qc

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], qnn: EstimatorQNN) -> np.ndarray:
    """
    Compute the Gram matrix between two datasets using the quantum kernel.
    """
    mat = []
    for x in a:
        row = []
        for y in b:
            val = qnn.predict([x, y])[0]
            row.append(val)
        mat.append(row)
    return np.array(mat)

__all__ = [
    "conv_circuit",
    "conv_layer",
    "pool_circuit",
    "pool_layer",
    "build_qcnn_ansatz",
    "build_qcnn_qnn",
    "quantum_kernel",
    "kernel_matrix",
]
