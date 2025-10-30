"""Quantum implementation of the hybrid QCNN.

The circuit mirrors the classical architecture while replacing each
convolutional and pooling block with a photonic‑style sub‑circuit
parameterised by beam‑splitter angles, squeezing and displacement
rotations.  These sub‑circuits are built from standard qiskit gates
and are composed into a depth‑reduced ansatz.  A ZFeatureMap feeds
the data into the circuit, and the observable is a single‑qubit Z
measurement on the first qubit.  The result is wrapped in an
EstimatorQNN for easy training with classical optimisers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def _photonic_conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Sub‑circuit mimicking a 2‑mode photonic layer using standard gates."""
    qc = QuantumCircuit(2)
    # Beam‑splitter angles as rotations
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def _photonic_pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Sub‑circuit for a pooling operation."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer composed of photonic sub‑circuits."""
    qc = QuantumCircuit(num_qubits, name="PhotonicConv")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = _photonic_conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = _photonic_conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Pooling layer composed of photonic sub‑circuits."""
    qc = QuantumCircuit(num_qubits, name="PhotonicPool")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _photonic_pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def _build_ansatz(num_qubits: int) -> QuantumCircuit:
    """Construct the full ansatz with alternating conv and pool layers."""
    ansatz = QuantumCircuit(num_qubits)
    # Feature map
    fm = ZFeatureMap(num_qubits)
    ansatz.compose(fm, range(num_qubits), inplace=True)
    # Layer sequence: conv → pool → conv → pool → conv → pool
    ansatz.compose(_conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(_pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2 + num_qubits // 2), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 2, "p2"), range(num_qubits // 2 + num_qubits // 2), inplace=True)
    ansatz.compose(_conv_layer(num_qubits // 4, "c3"), range(num_qubits // 4 + num_qubits // 4 + num_qubits // 4), inplace=True)
    ansatz.compose(_pool_layer(num_qubits // 4, "p3"), range(num_qubits // 4 + num_qubits // 4 + num_qubits // 4), inplace=True)
    return ansatz

def QCNNHybrid(num_qubits: int = 8) -> EstimatorQNN:
    """Return a quantum neural network that mirrors the hybrid QCNN."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    circuit = _build_ansatz(num_qubits)
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=ZFeatureMap(num_qubits).parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNHybrid"]
