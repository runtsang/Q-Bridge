"""Quantum classifier circuit builder with configurable entanglement and observables."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    entanglement: str = "linear",
    observables: List[str] | None = None,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational quantum circuit that mirrors the classical depth
    structure and offers flexible entanglement patterns.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (also the number of input features).
    depth : int
        Number of variational layers.
    entanglement : {"linear", "full"}, optional
        Entanglement scheme between qubits.  ``"linear"`` applies CZ gates
        between neighbours; ``"full"`` entangles every pair.
    observables : List[str] | None, optional
        Pauli strings to measure after the circuit.  If ``None`` a default
        list of single‑qubit Z observables is used.

    Returns
    -------
    circuit : QuantumCircuit
        The variational ansatz.
    encoding : Iterable
        Parameter vector for the data‑encoding rotations.
    weights : Iterable
        Parameter vector for the variational parameters.
    observables : List[SparsePauliOp]
        Pauli operators whose expectation values are to be measured.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for qubit, theta in zip(range(num_qubits), encoding):
        circuit.rx(theta, qubit)

    # Variational layers
    weight_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1

        # Entanglement
        if entanglement == "linear":
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cz(i, j)
        else:
            raise ValueError(f"Unsupported entanglement scheme: {entanglement}")

    # Observables
    if observables is None:
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
    else:
        observables = [SparsePauliOp(op) for op in observables]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
