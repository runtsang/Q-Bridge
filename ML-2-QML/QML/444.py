"""Quantum classifier with a ZZFeatureMap and RealAmplitudes ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Pauli, SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered quantum circuit combining a ZZFeatureMap and a RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Number of ansatz layers.

    Returns
    -------
    QuantumCircuit
        The full circuit ready for simulation or execution.
    Iterable
        List of encoding parameters (one per qubit).
    Iterable
        List of variational parameters (one per ansatz layer per qubit).
    list[SparsePauliOp]
        PauliZ observables for each qubit used as measurement targets.
    """
    # Feature map – data re‑uploading style
    encoding = ParameterVector("x", num_qubits)
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement="full", paulis="ZZ")

    # Ansatz – parameterized rotations with entanglement
    weights = ParameterVector("theta", num_qubits * depth)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=depth, entanglement="full", insert_barriers=False)

    # Build the circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.append(feature_map, range(num_qubits))
    circuit.append(ansatz, range(num_qubits))

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp(Pauli.from_label("I" * i + "Z" + "I" * (num_qubits - i - 1)))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
