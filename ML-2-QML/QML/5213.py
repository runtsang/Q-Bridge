"""Quantum QCNN‑style model using EstimatorQNN.

The circuit is built from ZFeatureMap, custom convolutional and pooling
layers, and a single‑qubit Pauli‑Z observable.  It is wrapped in an
EstimatorQNN object that can be trained by Qiskit Machine Learning
optimizers.  The public factory ``QCNNGen()`` returns the ready‑to‑use
quantum neural network.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in all convolutional layers."""
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


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer that couples adjacent qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for idx in range(0, num_qubits, 2):
        block = _conv_circuit(params[idx : idx + 3])
        qc.append(block, [idx, idx + 1])
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block that reduces entanglement."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Pool layer that maps a set of source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        block = _pool_circuit(params[:3])
        qc.append(block, [src, sink])
        params = params[3:]
    return qc


def QCNNGen() -> EstimatorQNN:
    """
    Factory creating a quantum neural network that implements a QCNN
    architecture.  The returned object is an :class:`EstimatorQNN` that
    can be used directly in Qiskit Machine Learning training loops.
    """
    # Feature map
    feature_map = ZFeatureMap(8)
    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First conv & pool
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second conv & pool
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Third conv & pool
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable (Pauli‑Z on the first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNGen", "QCNNGen"]
