"""Core circuit factory for the incremental data‑uploading classifier with optional feature‑encoding."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    The circuit now supports an optional data‑encoding layer that can be turned on
    or off via the *encoding* flag.
    """
    # --------------------------------------------------------------------------- #
    #  Parameter vectors
    # --------------------------------------------------------------------------- #
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    # --------------------------------------------------------------------------- #
    #  Circuit construction
    # --------------------------------------------------------------------------- #
    circuit = QuantumCircuit(num_qubits)

    # Optional classical encoding: first layer of Rx gates
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # entangling layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # --------------------------------------------------------------------------- #
    #  Observables
    # --------------------------------------------------------------------------- #
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
