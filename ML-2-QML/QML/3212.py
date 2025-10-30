"""Hybrid QCNN quantum neural network combining the original QCNN ansatz with a random feature‑map filter."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params):
    """Two‑qubit convolution block used in the QCNN ansatz."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


def pool_circuit(params):
    """Two‑qubit pooling block used in the QCNN ansatz."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def conv_layer(num_qubits, param_prefix):
    """Convolutional layer composed of disjoint two‑qubit blocks."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_layer(sources, sinks, param_prefix):
    """Pooling layer that discards qubits via two‑qubit blocks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


class HybridQCNN(EstimatorQNN):
    """Quantum neural network that augments the QCNN ansatz with a random feature‑map filter."""

    def __init__(self, shots: int = 1024, seed: int = 12345) -> None:
        algorithm_globals.random_seed = seed
        estimator = Estimator()

        # Feature map: Z‑feature map followed by a random circuit
        feature_map = ZFeatureMap(8)
        random_filter = random_circuit(8, 2)
        feature_map = feature_map.compose(random_filter, range(8))

        # QCNN ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

        # First Pooling Layer
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

        # Second Convolutional Layer
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

        # Second Pooling Layer
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Third Convolutional Layer
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

        # Third Pooling Layer
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        super().__init__(circuit=circuit.decompose(),
                         observables=observable,
                         input_params=feature_map.parameters,
                         weight_params=ansatz.parameters,
                         estimator=estimator)


def QCNN() -> HybridQCNN:
    """Return a configured :class:`HybridQCNN` instance."""
    return HybridQCNN()


__all__ = ["QCNN", "HybridQCNN"]
