"""
HybridQCNN: Quantum circuit backbone for feature extraction.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """2‑qubit convolution unitary used in QCNN layers."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def _pool_circuit(params):
    """2‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, prefix):
    """Construct a convolutional layer with 2‑qubit blocks."""
    qc = QuantumCircuit(num_qubits, name="conv_layer")
    params = ParameterVector(prefix, length=num_qubits//2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[idx:idx+3]), [q1, q2])
        idx += 3
    return qc

def _pool_layer(num_qubits, prefix):
    """Construct a pooling layer that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits, name="pool_layer")
    params = ParameterVector(prefix, length=num_qubits//2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_pool_circuit(params[idx:idx+3]), [q1, q2])
        idx += 3
    # Keep only the first half qubits
    qc.remove_bits(range(num_qubits//2, num_qubits))
    return qc

def _quantum_ansatz(num_qubits):
    """Variational ansatz that complements the QCNN layers."""
    qc = QuantumCircuit(num_qubits)
    qc.append(RealAmplitudes(num_qubits, reps=4), range(num_qubits))
    return qc

def HybridQCNNQNN(num_qubits: int = 8) -> EstimatorQNN:
    """
    Build a QCNN‑style circuit with convolution, pooling, and a variational ansatz,
    then wrap it in an EstimatorQNN for training.
    """
    # Feature map that embeds classical data
    feature_map = ZFeatureMap(num_qubits, reps=1, entanglement="full")
    # Build the QCNN circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, range(num_qubits), inplace=True)
    qc.compose(_conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    qc.compose(_pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)
    qc.compose(_conv_layer(num_qubits//2, "c2"), range(num_qubits//2), inplace=True)
    qc.compose(_pool_layer(num_qubits//2, "p2"), range(num_qubits//2), inplace=True)
    qc.compose(_quantum_ansatz(num_qubits//4), range(num_qubits//4), inplace=True)

    # Observable for a single‑qubit Z measurement on the remaining qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits//4 - 1), 1)])

    # Estimator for expectation values
    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=_conv_layer(num_qubits, "c1").parameters \
                    + _pool_layer(num_qubits, "p1").parameters \
                    + _conv_layer(num_qubits//2, "c2").parameters \
                    + _pool_layer(num_qubits//2, "p2").parameters \
                    + _quantum_ansatz(num_qubits//4).parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["HybridQCNNQNN", "HybridQCNNQNN"]
