"""Hybrid quantum classifier circuit factory with a hardware‑efficient ansatz.

The function returns:
    - circuit: QuantumCircuit object
    - encoding: list of ParameterVector objects (input encoding parameters)
    - weights: list of ParameterVector objects (variational parameters)
    - observables: list of SparsePauliOp objects to be measured
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int = 2,
    entanglement: str = "full",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a hardware‑efficient variational ansatz for classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (should match the number of input features).
    depth : int, default 2
        Number of variational layers.
    entanglement : str, {"full", "circular", "linear"}, default "full"
        Pattern of CNOT gates used for entanglement.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
        * circuit : QuantumCircuit
            The constructed quantum circuit.
        * encoding : Iterable
            List of ParameterVector objects for data encoding.
        * weights : Iterable
            List of ParameterVector objects for variational parameters.
        * observables : List[SparsePauliOp]
            Observables to measure (Pauli‑Z on each qubit).
    """
    # Data encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    weight_index = 0
    for _ in range(depth):
        # Variational rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_index], qubit)
            weight_index += 1

        # Entangling layer
        if entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cx(i, j)
        elif entanglement == "circular":
            for qubit in range(num_qubits):
                circuit.cx(qubit, (qubit + 1) % num_qubits)
        else:  # linear
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

    # Observables: Pauli-Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
