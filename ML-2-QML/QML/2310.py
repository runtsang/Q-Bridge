from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Single two‑qubit convolution block used by QCNN layers."""
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
    """Two‑qubit pooling block."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Convolutional layer over a register of qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(f"{prefix}_c", length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i // 2 * 3 : i // 2 * 3 + 3])
        qc.append(sub.to_instruction(), [i, i + 1])
    return qc

def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(f"{prefix}_p", length=len(sources) * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[idx * 3 : idx * 3 + 3])
        qc.append(sub.to_instruction(), [src, snk])
    return qc

def QCNN(num_layers: int = 3) -> EstimatorQNN:
    """
    Builds a QCNN ansatz with a feature map and a variable number of convolution‑pool layers.
    Returns an EstimatorQNN ready for training with a classical optimiser.
    """
    # Feature map for 8‑qubit input
    feature_map = ZFeatureMap(8)
    # Ansatz construction
    ansatz = QuantumCircuit(8, name="QCNN Ansatz")

    # Initial convolution and pooling
    ansatz.compose(_conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Additional layers
    for layer in range(2, num_layers + 1):
        # Convolution on the remaining qubits
        remaining = 8 // (2 ** (layer - 1))
        ansatz.compose(_conv_layer(remaining, f"c{layer}"), list(range(remaining)), inplace=True)
        # Pooling to halve the register
        sinks = list(range(remaining // 2))
        ansatz.compose(_pool_layer(list(range(remaining)), sinks, f"p{layer}"), list(range(remaining)), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable for classification (Z on first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Estimator for expectation values
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNN"]
