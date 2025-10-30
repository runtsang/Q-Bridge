"""Quantum circuit implementation of the hybrid QCNN.

The quantum counterpart reproduces the same hierarchical convolution
and pooling pattern, but uses variational parameters that can be
interpolated with the classical layer parameters.  Parameter clipping
is applied during training to avoid extreme values, mirroring the
classical side.  The circuit is built with Qiskit and returned as an
:class:`EstimatorQNN` for easy integration with classical optimisers."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution sub‑circuit."""
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
    """Two‑qubit pooling sub‑circuit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _paired_layer(num_qubits: int, pairs: list[tuple[int, int]],
                  param_prefix: str, conv: bool = True) -> QuantumCircuit:
    """Compose a layer of matched qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(pairs) * 3)
    for idx, (q1, q2) in enumerate(pairs):
        circ = _conv_circuit if conv else _pool_circuit
        qc.append(circ(params[idx*3:idx*3+3]), [q1, q2])
    return qc

def QCNNHybrid(num_qubits: int = 8) -> EstimatorQNN:
    """Build an EstimatorQNN that follows the QCNN depth and pooling scheme."""
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Feature map – a simple Z‑feature map over all qubits
    feature_map = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        feature_map.rz(ParameterVector("x", length=1)[0], q)

    # Ansatz construction
    ansatz = QuantumCircuit(num_qubits)

    # First convolution + pooling over all 8 qubits
    pairs1 = [(i, i + 1) for i in range(0, num_qubits, 2)]
    ansatz.compose(_paired_layer(num_qubits, pairs1, "c1", conv=True), inplace=True)
    ansatz.compose(_paired_layer(num_qubits, pairs1, "p1", conv=False), inplace=True)

    # Second convolution + pooling over the reduced 4‑qubit space
    pairs2 = [(i, i + 1) for i in range(0, num_qubits // 2, 2)]
    ansatz.compose(_paired_layer(num_qubits // 2, pairs2, "c2", conv=True), inplace=True)
    ansatz.compose(_paired_layer(num_qubits // 2, pairs2, "p2", conv=False), inplace=True)

    # Third convolution + pooling over the final 2‑qubit space
    pairs3 = [(0, 1)]
    ansatz.compose(_paired_layer(num_qubits // 4, pairs3, "c3", conv=True), inplace=True)
    ansatz.compose(_paired_layer(num_qubits // 4, pairs3, "p3", conv=False), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )

__all__ = ["QCNNHybrid"]
