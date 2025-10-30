"""Quantum classifier circuit with multi‑layer entanglement and
dynamic observable selection.

The function build_classifier_circuit returns a Qiskit QuantumCircuit,
a list of encoding parameters, a list of variational weight parameters,
and a list of SparsePauliOp observables. The ansatz includes a
parameterised data‑encoding layer followed by alternating rotation and
entangling layers. The observable set can be tuned for different
measurement strategies.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a variational quantum circuit for binary classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of entangling layers.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : Iterable
        List of ParameterVector objects for data encoding.
    weights : Iterable
        List of ParameterVector objects for variational parameters.
    observables : List[SparsePauliOp]
        Observable operators to measure post‑circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth * 2)  # two rotations per qubit per layer

    circ = QuantumCircuit(num_qubits)

    # Data encoding: RX on each qubit
    for qubit, param in enumerate(encoding):
        circ.rx(param, qubit)

    weight_index = 0
    for layer in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            circ.ry(weights[weight_index], qubit)
            weight_index += 1
            circ.rz(weights[weight_index], qubit)
            weight_index += 1
        # Entangling layer: a ring of CNOTs
        for qubit in range(num_qubits):
            circ.cx(qubit, (qubit + 1) % num_qubits)

    # Observables: single‑qubit Z on each qubit (default)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circ, list(encoding), list(weights), observables
