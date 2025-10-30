"""Quantum classifier that implements a hierarchical QCNN ansatz.

The circuit consists of:
  * A Z‑feature map that encodes the classical input into a superposition.
  * A stack of convolution‑plus‑pooling layers, each parameterised by a small
    set of rotation angles.  The depth of the stack is user‑configurable.
  * A single‑qubit observable (Z on the last remaining qubit) that yields
    a binary classification score.

The function returns the circuit, the encoding parameters, the weight
parameters, and the observable list, mirroring the signature of the
classical build_classifier_circuit.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Return a 2‑qubit convolution sub‑circuit parameterised by three angles."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[3 * (q // 2)], 0)
        sub.ry(params[3 * (q // 2) + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[3 * (q // 2) + 2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        qc.append(sub, [q, q + 1])
    return qc

def _pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Return a 2‑qubit pooling sub‑circuit parameterised by three angles."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q in range(0, num_qubits, 2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[3 * (q // 2)], 0)
        sub.ry(params[3 * (q // 2) + 1], 1)
        sub.cx(0, 1)
        sub.ry(params[3 * (q // 2) + 2], 1)
        qc.append(sub, [q, q + 1])
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a QCNN‑style quantum classifier and return metadata."""
    # Data encoding
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.ry(param, qubit)

    weight_params: List[ParameterVector] = []

    # Build hierarchical layers
    current_qubits = num_qubits
    for layer in range(depth):
        # Convolution
        conv = _conv_layer(current_qubits, f"c{layer}")
        circuit.append(conv, range(current_qubits))
        weight_params.append(ParameterVector(f"c{layer}", current_qubits * 3))

        # Pooling
        pool = _pool_layer(current_qubits // 2, f"p{layer}")
        circuit.append(pool, range(current_qubits // 2))
        weight_params.append(ParameterVector(f"p{layer}", (current_qubits // 2) * 3))

        # Reduce qubit count
        current_qubits //= 2

    # Observable on the last remaining qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (current_qubits - 1), 1)])
    return circuit, encoding, weight_params, [observable]

class QuantumClassifierModel:
    """Wrapper that builds the QCNN circuit and exposes an EstimatorQNN."""
    def __init__(self, num_qubits: int = 8, depth: int = 3) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def estimator_qnn(self) -> EstimatorQNN:
        estimator = Estimator()
        return EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=estimator,
        )
