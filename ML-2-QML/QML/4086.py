"""Hybrid sampler quantum implementation.

The quantum side mirrors the classical hybrid network:
  * a 2‑qubit sampler circuit with input and weight parameters
  * a QCNN‑style ansatz built from convolution and pooling layers
  * a feature map that encodes the 2‑dimensional input
  * an EstimatorQNN that evaluates the expectation value of a
    single‑qubit Z observable.

The circuit is constructed using Qiskit primitives and is ready
for execution on a simulator or real backend.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _sampler_circuit(inputs: ParameterVector, weights: ParameterVector) -> QuantumCircuit:
    """2‑qubit variational sampler circuit."""
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    return qc


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """3‑parameter convolution unit used in the QCNN ansatz."""
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
    """Builds a convolutional layer for an arbitrary number of qubits."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        layer = _conv_circuit(params[param_index:param_index+3])
        qc.append(layer.to_instruction(), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """3‑parameter pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that maps sources to sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for src, snk in zip(sources, sinks):
        layer = _pool_circuit(params[param_index:param_index+3])
        qc.append(layer.to_instruction(), [src, snk])
        qc.barrier()
        param_index += 3
    return qc


def _build_ansatz() -> QuantumCircuit:
    """Constructs the full QCNN ansatz."""
    ansatz = QuantumCircuit(8)

    # First convolution
    ansatz.compose(_conv_layer(8, "c1"), [i for i in range(8)], inplace=True)
    # First pooling
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), [i for i in range(8)], inplace=True)
    # Second convolution
    ansatz.compose(_conv_layer(4, "c2"), [i for i in range(4, 8)], inplace=True)
    # Second pooling
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), [i for i in range(4, 8)], inplace=True)
    # Third convolution
    ansatz.compose(_conv_layer(2, "c3"), [i for i in range(6, 8)], inplace=True)
    # Third pooling
    ansatz.compose(_pool_layer([0], [1], "p3"), [i for i in range(6, 8)], inplace=True)

    return ansatz


def SamplerQNN() -> EstimatorQNN:
    """Factory returning a quantum EstimatorQNN representing the hybrid sampler."""
    # Feature map for 2‑dimensional input
    feature_map = ZFeatureMap(2)

    # Sampler parameters
    inputs = ParameterVector("input", length=2)
    weights = ParameterVector("weight", length=4)

    # Sampler circuit
    sampler_circ = _sampler_circuit(inputs, weights)

    # Full ansatz
    ansatz = _build_ansatz()

    # Combine feature map, sampler, and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(sampler_circ, [0, 1], inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable (single‑qubit Z on qubit 0)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=weights + ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["SamplerQNN"]
