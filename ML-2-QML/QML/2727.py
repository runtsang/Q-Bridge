"""Quantum autoencoder built from QCNN layers using an EstimatorQNN."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in QCNN."""
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
    """Two‑qubit pooling block used in QCNN."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer with pairwise conv_circuits."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        param_index += 3
    return qc


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_index:param_index + 3]), [source, sink])
        param_index += 3
    return qc


def QCNNAutoencoder() -> EstimatorQNN:
    """Quantum autoencoder that mirrors the classical QCNN architecture."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map to embed classical data
    feature_map = ZFeatureMap(8)
    feature_map.decompose()

    # Build the encoder (convolution + pooling)
    encoder = QuantumCircuit(8)
    encoder.compose(_conv_layer(8, "c1"), inplace=True)
    encoder.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    encoder.compose(_conv_layer(4, "c2"), inplace=True)
    encoder.compose(_pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Decoder mirrors the encoder but in reverse order
    decoder = QuantumCircuit(8)
    decoder.compose(_pool_layer([0, 1], [2, 3], "p2d"), inplace=True)
    decoder.compose(_conv_layer(4, "c2d"), inplace=True)
    decoder.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1d"), inplace=True)
    decoder.compose(_conv_layer(8, "c1d"), inplace=True)

    # Full circuit: input -> encoder -> decoder
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(encoder, inplace=True)
    circuit.compose(decoder, inplace=True)

    # Observable for reconstruction loss (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Combine input and weight parameters
    input_params = feature_map.parameters
    weight_params = encoder.parameters + decoder.parameters

    # Interpret returns the expectation value as a scalar
    def interpret(x):
        return x

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNAutoencoder"]
