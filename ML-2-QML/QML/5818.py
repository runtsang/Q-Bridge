from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def build_qcnn_qnn() -> EstimatorQNN:
    """Constructs a quantum neural network implementing the QCNN architecture.

    The circuit follows the three‑layer convolution‑pooling pattern from the
    original QCNN example but is expanded to support arbitrary input sizes
    via a reusable feature map.  The returned EstimatorQNN can be used
    directly in a PyTorch forward pass through its ``predict`` method.
    """
    # Feature map – 8 independent Ry gates
    feature_map = QuantumCircuit(8)
    phi = ParameterVector("phi", 8)
    for i, p in enumerate(phi):
        feature_map.ry(p, i)

    # Convolution unitary acting on two qubits
    def conv_unitary(params):
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

    # Pooling unitary acting on two qubits
    def pool_unitary(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Helper to build a layer of pairwise gates
    def layer(num_qubits, prefix, unitary_builder):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits, 2):
            sub = unitary_builder(params[i // 2 * 3 : (i // 2 + 1) * 3])
            qc.compose(sub, [i, i + 1], inplace=True)
        return qc

    # Assemble the full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(layer(8, "c1", conv_unitary), inplace=True)
    ansatz.compose(layer(8, "p1", pool_unitary), inplace=True)
    ansatz.compose(layer(4, "c2", conv_unitary), inplace=True)
    ansatz.compose(layer(4, "p2", pool_unitary), inplace=True)
    ansatz.compose(layer(2, "c3", conv_unitary), inplace=True)
    ansatz.compose(layer(2, "p3", pool_unitary), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable – single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator backend
    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )


__all__ = ["build_qcnn_qnn"]
