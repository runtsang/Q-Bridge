"""Quantum classifier factory with configurable entanglement and rotation gates."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    encoding: str = "rx",
    rotation: str = "ry",
    entanglement: str = "full",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a parameter‑ized variational circuit for binary classification.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    encoding : str, optional
        Gate used for feature encoding ('rx', 'ry', 'rz', or 'ryz').
    rotation : str, optional
        Gate used for variational parameters ('ry', 'rz', 'ryz').
    entanglement : str, optional
        Pattern of two‑qubit entanglement ('full', 'circular', or 'none').

    Returns
    -------
    circuit : QuantumCircuit
        The assembled quantum circuit.
    encoding_params : Iterable
        ParameterVector for feature encoding.
    weights : Iterable
        ParameterVector for variational parameters.
    observables : list[SparsePauliOp]
        Pauli‑Z measurement operators for each qubit.
    """
    enc_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Feature encoding
    for qubit, param in enumerate(enc_params):
        if encoding == "rx":
            circuit.rx(param, qubit)
        elif encoding == "ry":
            circuit.ry(param, qubit)
        elif encoding == "rz":
            circuit.rz(param, qubit)
        else:  # ryz
            circuit.ry(param, qubit)
            circuit.rz(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            if rotation == "ry":
                circuit.ry(weight_params[idx], qubit)
            elif rotation == "rz":
                circuit.rz(weight_params[idx], qubit)
            else:  # ryz
                circuit.ry(weight_params[idx], qubit)
                circuit.rz(weight_params[idx], qubit)
            idx += 1

        # Entanglement
        if entanglement == "full":
            for q in range(num_qubits):
                circuit.cx(q, (q + 1) % num_qubits)
        elif entanglement == "circular":
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
        # 'none' introduces no entanglement

    # Observable operators (Pauli‑Z on each qubit)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(enc_params), list(weight_params), observables


__all__ = ["build_classifier_circuit"]
