"""Core circuit factory for the incremental data‑uploading classifier, extended with flexible entanglement and feature‑map depth."""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int,
                             entanglement: str = "cyclic",
                             feature_map_depth: int = 1) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    entanglement : str
        Entanglement pattern: ``"cyclic"``, ``"full"``, or ``"none"``.
    feature_map_depth : int
        Depth of the initial feature‑map encoding (multiple RX layers per qubit).

    Returns
    -------
    circuit : QuantumCircuit
        Prepared quantum circuit.
    encoding : Iterable[ParameterVector]
        Parameter vector for classical feature encoding.
    weights : Iterable[ParameterVector]
        Parameter vector for variational parameters.
    observables : list[SparsePauliOp]
        Z‑observables used to read out class logits.
    """
    # Feature‑map encoding
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)

    # Layered feature map (multiple RX per qubit)
    for _ in range(feature_map_depth):
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

    # Create variational weights
    weights = ParameterVector("theta", num_qubits * depth)

    # Variational layers with configurable entanglement
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1

        if entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cz(i, j)
        elif entanglement == "cyclic":
            for i in range(num_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(num_qubits - 1, 0)
        # ``none`` leaves the qubits unentangled

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
