"""Quantum QCNN with parameter‑shared convolution/pooling and depthwise rotations."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Set reproducible randomness
algorithm_globals.random_seed = 12345


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Convolution unitary shared across all qubit pairs."""
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
    """Pooling unitary shared across all qubit pairs."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _depthwise_rotations(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Depthwise rotation block to aid efficient simulation."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc


def _conv_layer(num_qubits: int, conv_params: ParameterVector) -> QuantumCircuit:
    """Apply shared convolution unitary to all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(conv_params), [q1, q2])
        qc.barrier()
    return qc


def _pool_layer(num_qubits: int, pool_params: ParameterVector) -> QuantumCircuit:
    """Apply shared pooling unitary to all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_pool_circuit(pool_params), [q1, q2])
        qc.barrier()
    return qc


def QCNNEnhanced() -> EstimatorQNN:
    """Return a parameter‑shared QCNN EstimatorQNN with depthwise rotations."""
    estimator = StatevectorEstimator()

    # Feature map
    feature_map = ZFeatureMap(8)
    feature_map.decompose().draw("mpl", style="clifford")

    # Shared parameters
    conv_params = ParameterVector("θ_c", length=3)
    pool_params = ParameterVector("θ_p", length=3)

    # Build ansatz with shared conv/pool layers
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution + depthwise rotation
    ansatz.append(_conv_layer(8, conv_params), range(8))
    ansatz.append(_depthwise_rotations(8, "dw1"), range(8))

    # First pooling
    ansatz.append(_pool_layer(8, pool_params), range(8))

    # Second convolution + depthwise rotation
    ansatz.append(_conv_layer(4, conv_params), range(4, 8))
    ansatz.append(_depthwise_rotations(4, "dw2"), range(4, 8))

    # Second pooling
    ansatz.append(_pool_layer(4, pool_params), range(4, 8))

    # Third convolution + depthwise rotation
    ansatz.append(_conv_layer(2, conv_params), range(6, 8))
    ansatz.append(_depthwise_rotations(2, "dw3"), range(6, 8))

    # Third pooling
    ansatz.append(_pool_layer(2, pool_params), range(6, 8))

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(ansatz, range(8))

    # Observable: weighted sum of Z on first and last qubit to emulate residual
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 0.5), ("I" * 7 + "Z", 0.5)])

    # Construct EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


__all__ = ["QCNNEnhanced"]
