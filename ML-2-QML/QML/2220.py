"""Quantum circuit factory for hybrid classifier."""
from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data‑encoding and variational parameters
    that matches the classical metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the number of classical features fed to the quantum head.
    depth : int
        Depth of the variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The parameterised circuit.
    encoding : List[ParameterVector]
        Parameter vectors used for data encoding.
    weights : List[ParameterVector]
        Parameter vectors used for the variational layers.
    observables : List[SparsePauliOp]
        Pauli‑Z observables on each qubit, matching the classical sigmoid output.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX gates
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling CZ chain
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit"]
