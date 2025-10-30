"""Quantum QCNN with depth‑controlled tree ansatz and shared feature map."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit as in the original QCNN."""
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Apply the convolution unit to every adjacent pair of qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    # Wrap the sub‑circuit as an instruction
    inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(inst, qubits)


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Apply the pooling unit to each source‑sink pair."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, snk in zip(sources, sinks):
        qc = qc.compose(
            pool_circuit(params[param_index : param_index + 3]),
            [src, snk],
        )
        qc.barrier()
        param_index += 3
    inst = qc.to_instruction()
    return QuantumCircuit(num_qubits).append(inst, range(num_qubits))


def build_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Construct a depth‑controlled tree‑structured ansatz."""
    ansatz = QuantumCircuit(num_qubits)
    # Define pooling patterns that match the original QCNN
    pooling_patterns = [
        ([0, 1, 2, 3], [4, 5, 6, 7]),
        ([0, 1], [2, 3]),
        ([0], [1]),
    ]
    for d in range(depth):
        # Convolution layer
        conv = conv_layer(num_qubits, f"c{d}")
        ansatz.compose(conv, range(num_qubits), inplace=True)
        # Pooling layer – reuse the last pattern if depth is large
        sources, sinks = pooling_patterns[min(d, len(pooling_patterns) - 1)]
        pool = pool_layer(sources, sinks, f"p{d}")
        ansatz.compose(pool, range(num_qubits), inplace=True)
    return ansatz


class QCNNEnhanced(EstimatorQNN):
    """
    Quantum QCNN with a configurable depth.  It inherits from
    :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN` and
    exposes the same API while internally building a tree‑structured
    ansatz that mirrors the classical branching logic.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3, **kwargs) -> None:
        # Feature map
        feature_map = ZFeatureMap(num_qubits)
        # Build the ansatz
        ansatz = build_ansatz(num_qubits, depth)
        # Observable – single‑qubit Z on the first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        # Estimator primitive
        estimator = Estimator()
        # Call the parent constructor
        super().__init__(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
            **kwargs,
        )

    @classmethod
    def build(cls, num_qubits: int = 8, depth: int = 3, **kwargs) -> "QCNNEnhanced":
        """Convenience constructor returning a ready‑to‑use instance."""
        return cls(num_qubits=num_qubits, depth=depth, **kwargs)


__all__ = ["QCNNEnhanced"]
