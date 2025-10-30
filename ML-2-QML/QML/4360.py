"""Quantum QCNN with a stochastic random layer and all‑qubit measurement.

The circuit builds upon the convolution‑pooling ansatz from the original QCNN
seed, but augments it with a Qiskit random circuit on a subset of wires to
inject stochasticity (inspired by the RandomLayer in Quantum‑NAT).  The
resulting EstimatorQNN can be trained with any classical optimiser.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Convolution and pooling primitives
# --------------------------------------------------------------------------- #
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
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _conv_circuit(params[param_index : param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc


def _pool_layer(sources, sinks, param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        sub = _pool_circuit(params[param_index : param_index + 3])
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc


# --------------------------------------------------------------------------- #
# Full QCNN ansatz with a stochastic random layer
# --------------------------------------------------------------------------- #
def QCNN() -> EstimatorQNN:
    """Return a qiskit EstimatorQNN that mirrors the hybrid QCNN architecture."""
    # Feature map
    feature_map = ZFeatureMap(8)
    # Ansatz construction
    ansatz = QuantumCircuit(8, name="QCNN Ansatz")

    # First convolutional layer
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First pooling layer
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Stochastic random layer on 4 wires (inspired by Quantum‑NAT)
    rand_layer = random_circuit(4, 3, seed=42, skip_last=True).to_instruction()
    ansatz.append(rand_layer, [4, 5, 6, 7], inplace=True)

    # Second convolutional layer
    ansatz.compose(_conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second pooling layer
    ansatz.compose(_pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolutional layer
    ansatz.compose(_conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third pooling layer
    ansatz.compose(_pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable: measure all qubits in Z basis
    observable = SparsePauliOp.from_list([("Z" * 8, 1)])

    # Estimator for expectation values
    estimator = Estimator()

    # Build EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNN"]
