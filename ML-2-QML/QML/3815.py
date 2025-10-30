"""Hybrid quantum autoencoder based on a QCNN circuit and a swap‑test decoder."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def _identity_interpret(x):
    """Return raw measurement results."""
    return x


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit used in the QCNN."""
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
    """Apply parallel convolution blocks."""
    qc = QuantumCircuit(num_qubits)
    param_vec = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(_conv_circuit(param_vec[i:i + 3]), [i, i + 1])
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Apply pooling between source and sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_vec = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk, param in zip(sources, sinks, param_vec):
        qc.append(_pool_circuit(param[:3]), [src, snk])
    return qc


def HybridAutoencoder() -> SamplerQNN:
    """Return a quantum autoencoder QNN."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Build ansatz: QCNN style encoding
    ansatz = QuantumCircuit(8, name="QCNN_Encoder")
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(list(range(4)), list(range(4, 8)), "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), inplace=True)

    # Full circuit: feature map -> ansatz -> swap‑test decoder
    circuit = QuantumCircuit(9, 1)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    circuit.h(8)
    for i in range(2):
        circuit.cswap(8, i, i + 4)
    circuit.h(8)
    circuit.measure(8, 0)

    qnn = SamplerQNN(
        circuit=circuit.decompose(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=_identity_interpret,
        output_shape=(2,),
        sampler=sampler,
    )
    return qnn


__all__ = ["HybridAutoencoder"]
