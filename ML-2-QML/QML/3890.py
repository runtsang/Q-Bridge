"""
Hybrid quantum autoencoder that composes a QCNN ansatz with a swap-test based
reconstruction circuit.  The circuit encodes classical input into a
latent subspace and decodes it back, providing a fully quantum
implementation of the hybrid concept.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two-qubit convolution unit used by the QCNN ansatz."""
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
    """Two-qubit pooling unit used by the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Build a convolutional layer over `num_qubits`."""
    qc = QuantumCircuit(num_qubits)
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


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Build a pooling layer that merges `sources` into `sinks`."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        sub = _pool_circuit(params[param_index : param_index + 3])
        qc.append(sub, [source, sink])
        qc.barrier()
        param_index += 3
    return qc


class HybridAutoencoderQNN(EstimatorQNN):
    """A quantum autoencoder that uses a QCNN ansatz for encoding and a swap-test for reconstruction."""

    def __init__(self, num_qubits: int = 8):
        algorithm_globals.random_seed = 42
        estimator = Estimator()

        # QCNN feature map
        feature_map = ZFeatureMap(num_qubits)

        # Build QCNN ansatz progressively
        ansatz = QuantumCircuit(num_qubits)

        # First convolutional + pooling stage
        ansatz.compose(_conv_layer(num_qubits, "c1"), inplace=True)
        ansatz.compose(_pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)

        # Second convolutional + pooling stage (down to half qubits)
        reduced = num_qubits // 2
        ansatz.compose(_conv_layer(reduced, "c2"), inplace=True)
        ansatz.compose(
            _pool_layer(list(range(reduced // 2)), list(range(reduced // 2, reduced)), "p2"),
            inplace=True,
        )

        # Third convolutional + pooling stage (down to single qubit)
        final = reduced // 2
        ansatz.compose(_conv_layer(final, "c3"), inplace=True)
        ansatz.compose(
            _pool_layer([0], [1], "p3"),
            inplace=True,
        )

        # Swap-test reconstruction: auxiliary qubit for measuring overlap
        aux = QuantumRegister(1, "aux")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(aux, cr)
        qc.h(aux[0])
        qc.x(aux[0])
        for q in range(num_qubits):
            qc.cswap(aux[0], q, q + 1) if q + 1 < num_qubits else None
        qc.h(aux[0])
        qc.measure(aux[0], cr[0])

        # Combine feature map, ansatz, and reconstruction
        full_circuit = QuantumCircuit(num_qubits + 1)
        full_circuit.compose(feature_map, range(num_qubits), inplace=True)
        full_circuit.compose(ansatz, range(num_qubits), inplace=True)
        full_circuit.compose(qc, [num_qubits], inplace=True)

        # Observable for reconstruction (Z on auxiliary qubit)
        observable = SparsePauliOp.from_list([("Z", 1)])

        super().__init__(
            circuit=full_circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )


def HybridAutoencoder(num_qubits: int = 8) -> HybridAutoencoderQNN:
    """Factory function returning the configured hybrid quantum autoencoder."""
    return HybridAutoencoderQNN(num_qubits)


__all__ = ["HybridAutoencoderQNN", "HybridAutoencoder"]
