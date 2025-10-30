"""Hybrid quantum estimator that merges QCNN ansatz with a quantum fully‑connected layer.

The circuit is built in three stages:
    1. A Z‑feature map encodes the classical input.
    2. A QCNN‑style ansatz (convolution + pooling) acts on 8 qubits.
    3. A separate 4‑qubit quantum fully‑connected sub‑circuit processes the pooled representation.

The estimator uses Qiskit’s StatevectorEstimator to compute the expectation of a Z observable on the final qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# --- Helper circuits ---------------------------------------------------------
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Three‑parameter convolution unit used in QCNN."""
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
    """Three‑parameter pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _layer(circuit_fn, num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Generic layer that composes the given unit on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        unit = circuit_fn(params[idx:idx+3])
        qc.append(unit, [q1, q2])
        qc.barrier()
        idx += 3
    return qc

# --- QCNN ansatz -------------------------------------------------------------
def _qcnn_ansatz() -> QuantumCircuit:
    """Full QCNN ansatz: 3 conv layers + 3 pool layers on 8 qubits."""
    qc = QuantumCircuit(8)

    # Convolutional layers
    qc.append(_layer(_conv_circuit, 8, "c1"), range(8))
    qc.append(_layer(_conv_circuit, 4, "c2"), range(4, 8))
    qc.append(_layer(_conv_circuit, 2, "c3"), range(6, 8))

    # Pooling layers
    qc.append(_layer(_pool_circuit, 8, "p1"), range(8))
    qc.append(_layer(_pool_circuit, 4, "p2"), range(4, 8))
    qc.append(_layer(_pool_circuit, 2, "p3"), range(6, 8))

    return qc

# --- Quantum fully‑connected sub‑circuit -------------------------------------
def _qfc_circuit() -> QuantumCircuit:
    """4‑qubit circuit performing random layers + RX/RZ/RY/CRX."""
    qc = QuantumCircuit(4)
    # Random layer
    for _ in range(50):
        qc.rx(np.random.uniform(0, 2*np.pi), 0)
        qc.ry(np.random.uniform(0, 2*np.pi), 1)
        qc.rz(np.random.uniform(0, 2*np.pi), 2)
        qc.cx(0, 3)
    # Parameterised gates
    theta = ParameterVector("qfc", length=4)
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.rz(theta[2], 2)
    qc.crx(theta[3], 0, 2)
    return qc

# --- Hybrid estimator --------------------------------------------------------
def HybridQuantumEstimator() -> EstimatorQNN:
    """
    Builds a hybrid QNN that concatenates a QCNN ansatz with a 4‑qubit quantum
    fully‑connected layer.  The final observable is a Z on the last qubit.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map (8‑qubit Z‑map)
    feature_map = QuantumCircuit(8)
    for q in range(8):
        feature_map.rz(ParameterVector("phi", 8)[q], q)

    # Assemble full circuit
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(_qcnn_ansatz(), range(8))
    circuit.append(_qfc_circuit(), range(4))  # only first 4 qubits

    # Observable on the last qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=(_qcnn_ansatz().parameters + _qfc_circuit().parameters),
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQuantumEstimator"]
