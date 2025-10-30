"""
QCNNHybridQNN: Quantum circuit that emulates the QCNN‑style architecture.

The ansatz mirrors the layered convolution‑pool structure from the QCNN seed,
while the EstimatorQNN wrapper follows the EstimatorQNN example.  The
resulting object can be trained with classical optimizers such as COBYLA or
Adam.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def QCNNHybridQNN() -> EstimatorQNN:
    """Builds and returns a QCNN‑style EstimatorQNN instance."""
    # Feature map
    feature_map = ZFeatureMap(8)
    feature_map.decompose()  # clean up for readability

    # ----- Convolution block (2 qubits) -----
    def conv_block(params: ParameterVector) -> QuantumCircuit:
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

    # ----- Pooling block (2 qubits) -----
    def pool_block(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Helper to create a layer of blocks
    def layer(block_fn, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(block_fn(params[i:i + 3]), [i, i + 1])
        return qc

    # Assemble the ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(layer(conv_block, 8, "c1"), inplace=True)
    ansatz.compose(layer(pool_block, 8, "p1"), inplace=True)
    ansatz.compose(layer(conv_block, 4, "c2"), inplace=True)
    ansatz.compose(layer(pool_block, 4, "p2"), inplace=True)
    ansatz.compose(layer(conv_block, 2, "c3"), inplace=True)
    ansatz.compose(layer(pool_block, 2, "p3"), inplace=True)

    # Full circuit: feature map + ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for regression (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap in EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=Estimator(),
    )
    return qnn


__all__ = ["QCNNHybridQNN"]
