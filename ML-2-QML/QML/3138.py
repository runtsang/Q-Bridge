"""Quantum implementation of QCNNGraphHybrid.

The class QCNNGraphHybrid encapsulates a variational quantum circuit that
mirrors the classical graph‑based convolution pattern.  It exposes a
`predict` method that accepts a batch of input parameters and returns
the expectation value of the chosen observable.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Convolution and pooling primitives
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i:i+3])
        qc.append(sub, [i, i+1])
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (s, t) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[i*3:i*3+3])
        qc.append(sub, [s, t])
    return qc

class QCNNGraphHybrid:
    """Quantum-only implementation of the hybrid QCNN.

    The circuit follows the same convolution–pooling pattern as the
    classical‑quantum hybrid version, but all processing is carried out
    in a variational quantum circuit.  The class offers a `predict`
    method that can be used directly for inference or as a layer in a
    larger quantum machine‑learning pipeline.
    """
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        # Feature map parameters
        self.feature_map = ParameterVector("φ", length=num_qubits)

        # Build ansatz
        self.circuit = QuantumCircuit(num_qubits)
        # First conv + pool
        self.circuit.compose(_conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
        self.circuit.compose(_pool_layer(list(range(0, num_qubits, 2)),
                                        list(range(1, num_qubits, 2)), "p1"),
                            list(range(num_qubits)), inplace=True)
        # Second conv + pool on half qubits
        half = num_qubits // 2
        self.circuit.compose(_conv_layer(half, "c2"), list(range(half)), inplace=True)
        self.circuit.compose(_pool_layer(list(range(half)), [half], "p2"),
                            list(range(half)), inplace=True)

        # Observable
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the expectation value for each input sample."""
        return self.qnn.predict(inputs)

__all__ = ["QCNNGraphHybrid"]
