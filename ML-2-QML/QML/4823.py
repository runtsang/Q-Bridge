"""Quantum implementation of the QCNN architecture.

The quantum version follows the original conv‑pool construction but augments
it with an additional encoding circuit and a depth‑controlled variational
ansatz.  The ansatz is built by composing the conv_layer and pool_layer
functions and then appending a classifier‑style trainable block as in
QuantumClassifierModel.  The circuit is turned into an EstimatorQNN for
training with the COBYLA optimiser.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block used in the conv_layer."""
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

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, name: str = "conv") -> QuantumCircuit:
    """Construct a convolutional layer acting on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    param_prefix = f"{name}_"
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        block = conv_circuit(params[idx:idx+3])
        qc.append(block, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def pool_layer(num_qubits: int, name: str = "pool") -> QuantumCircuit:
    """Construct a pooling layer acting on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    param_prefix = f"{name}_"
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        block = pool_circuit(params[idx:idx+3])
        qc.append(block, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

def build_classifier_ansatz(num_qubits: int, depth: int):
    """Append a classifier‑style variational block to the ansatz."""
    encoding = ParameterVector("x", num_qubits)
    weights   = ParameterVector("theta", num_qubits * depth)

    ansatz = QuantumCircuit(num_qubits)
    # Feature encoding
    for q, param in zip(range(num_qubits), encoding):
        ansatz.rx(param, q)

    # Depth‑controlled variational layers
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            ansatz.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            ansatz.cz(q, q + 1)

    return ansatz, list(encoding), list(weights)

def QCNN() -> EstimatorQNN:
    """Return a fully‑defined EstimatorQNN that implements the QCNN."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()
    num_qubits = 8

    # Build the main ansatz with conv‑pool layers
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(conv_layer(num_qubits), inplace=True)
    ansatz.compose(pool_layer(num_qubits), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 2), inplace=True)
    ansatz.compose(pool_layer(num_qubits // 2), inplace=True)
    ansatz.compose(conv_layer(num_qubits // 4), inplace=True)
    ansatz.compose(pool_layer(num_qubits // 4), inplace=True)

    # Append the classifier ansatz
    clf_ansatz, encoding_params, weight_params = build_classifier_ansatz(num_qubits, depth=2)
    ansatz.append(clf_ansatz, range(num_qubits))

    # Feature map
    feature_map = ZFeatureMap(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observable: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN"]
