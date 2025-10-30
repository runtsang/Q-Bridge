"""Quantum implementation of a QCNN with a fully‑connected layer.

The circuit combines the convolution and pooling layers from the
original QCNN example with an additional fully‑connected block
implemented as a parameterized Ry circuit.  The output is a
single‑qubit expectation value that can be used as a classification
score.  The design keeps the same interface as the original QCNN
factory while adding the additional layer, facilitating direct
comparison with the classical implementation above.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals
from typing import Iterable

algorithm_globals.random_seed = 12345
_ESTIMATOR = StatevectorEstimator()

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

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="conv")
    qubits = list(range(num_qubits))
    params = ParameterVector(prefix, length=num_qubits * 3)
    idx = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    for q1, q2 in zip(qubits[1::2], (qubits[2::2] + [0])):
        qc.append(_conv_circuit(params[idx:idx+3]), [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def _pool_layer(sources: Iterable[int], sinks: Iterable[int], prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="pool")
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(_pool_circuit(params[:3]), [src, snk])
        qc.barrier()
        params = params[3:]
    return qc

def _fully_connected_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """A simple fully connected layer implemented as Ry rotations on each qubit."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits)
    for q, p in zip(range(num_qubits), params):
        qc.ry(p, q)
    return qc

def QCNNFCL() -> EstimatorQNN:
    """Builds a QCNN with an added fully‑connected layer and returns an EstimatorQNN."""
    feature_map = ZFeatureMap(8)
    # Build ansatz
    ansatz = QuantumCircuit(8, name="ansatz")
    # First conv + pool
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8), inplace=True)
    # Second conv + pool
    ansatz.compose(_conv_layer(4, "c2"), range(4,8), inplace=True)
    ansatz.compose(_pool_layer([0,1], [2,3], "p2"), range(4,8), inplace=True)
    # Third conv + pool
    ansatz.compose(_conv_layer(2, "c3"), range(6,8), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), range(6,8), inplace=True)
    # Fully connected layer on the remaining two qubits
    ansatz.compose(_fully_connected_layer(2, "fc"), range(6,8), inplace=True)

    # Combine with feature map
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=_ESTIMATOR,
    )
    return qnn

__all__ = ["QCNNFCL", "QCNNFCL"]
