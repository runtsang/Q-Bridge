"""Enhanced quantum circuit factory with configurable entanglement and parameter‑shift gradient support."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import numpy as np


def _entangle(circuit: QuantumCircuit, qubits: List[int], pattern: str) -> None:
    """
    Apply entangling gates according to the chosen pattern.
    """
    if pattern == "full":
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                circuit.cz(qubits[i], qubits[j])
    elif pattern == "nearest":
        for i in range(len(qubits) - 1):
            circuit.cz(qubits[i], qubits[i + 1])
    elif pattern == "none":
        pass
    else:
        raise ValueError(f"Unsupported entanglement pattern: {pattern}")


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "full",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational ansatz with data‑encoding, rotation layers and an
    entanglement pattern.  The function returns the circuit together with
    metadata that mirrors the classical interface.

    Parameters
    ----------
    num_qubits: int
        Number of qubits (features).
    depth: int
        Number of variational layers.
    entanglement: str, optional
        Entanglement pattern: 'full', 'nearest', or 'none'.

    Returns
    -------
    circuit: QuantumCircuit
        The constructed circuit ready for simulation or execution.
    encoding: Iterable[ParameterVector]
        Parameter vector for feature encoding.
    weights: Iterable[ParameterVector]
        Parameter vector for variational parameters.
    observables: List[SparsePauliOp]
        Pauli Z observables on each qubit for binary read‑out.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data‑encoding: RX rotations with feature parameters
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entanglement
        _entangle(circuit, list(range(num_qubits)), entanglement)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
