"""Quantum circuit builder for a QCNN‑style classifier with data‑uploading encoding."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in all conv layers."""
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
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Compose convolutional layers across neighbouring qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    idx = 0
    params = ParameterVector(prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def _pool_layer(sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
    """Compose pooling layers across specified source/sink pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    idx = 0
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for src, sink in zip(sources, sinks):
        sub = _pool_circuit(params[idx : idx + 3])
        qc.append(sub, [src, sink])
        qc.barrier()
        idx += 3
    return qc


def build_classifier_circuit(num_qubits: int, depth: int = 3) -> Tuple[
    QuantumCircuit, Iterable[ParameterVector], List[int], List[SparsePauliOp]
]:
    """
    Construct a layered QCNN circuit with data‑uploading encoding.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) to encode.
    depth : int
        Number of convolution–pooling pairs; defaults to 3 to match the classical QCNN.

    Returns
    -------
    circuit : QuantumCircuit
        Full variational circuit ready for use with EstimatorQNN.
    encoding : Iterable[ParameterVector]
        Parameters used for the initial data‑encoding Rx gates.
    weight_sizes : List[int]
        Count of trainable parameters per variational block.
    observables : List[SparsePauliOp]
        Measurement operators for classification.
    """
    # Data‑encoding layer
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth * 3)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Build the feature‑map (ZFeatureMap) followed by the QCNN ansatz
    feature_map = ZFeatureMap(num_qubits)
    circuit.compose(feature_map, list(range(num_qubits)), inplace=True)

    # Ansatz construction: alternating conv and pool layers
    for i in range(depth):
        conv = _conv_layer(num_qubits, f"c{i}")
        pool = _pool_layer(list(range(num_qubits)), list(range(num_qubits)), f"p{i}")
        circuit.compose(conv, range(num_qubits), inplace=True)
        circuit.compose(pool, range(num_qubits), inplace=True)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1 - i), 1)]) for i in range(num_qubits)
    ]

    # Weight sizes: total number of trainable parameters per conv/pool block
    weight_sizes = [3 * num_qubits] * (2 * depth)

    return circuit, [encoding], weight_sizes, observables


def QCNN_estimator() -> EstimatorQNN:
    """
    Convenience factory that returns a fully constructed EstimatorQNN for the QCNN.

    The circuit is built with a fixed depth of 3 and 8 qubits, matching the classical model.
    """
    circuit, encoding, weight_sizes, observables = build_classifier_circuit(8, depth=3)
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observables,
        input_params=encoding[0],
        weight_params=ParameterVector("theta", len(weight_sizes) * 8),
        estimator=estimator,
    )


__all__ = ["build_classifier_circuit", "QCNN_estimator"]
