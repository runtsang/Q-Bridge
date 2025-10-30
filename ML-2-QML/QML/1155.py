"""Core circuit factory for the incremental data‑uploading classifier with custom measurement bases."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    encoding_gate: str = "rx",
    measurement_bases: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[PauliSumOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Allows custom encoding gate and per‑qubit measurement bases.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    encoding_gate : str, optional
        Gate used for data encoding. Supported: 'rx', 'ry', 'rz'.
    measurement_bases : list[str], optional
        Measurement basis for each qubit. If ``None`` defaults to ['Z'] * num_qubits.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.
    Iterable
        List of encoding parameters.
    Iterable
        List of variational parameters.
    List[PauliSumOp]
        Measurement operators.
    """
    np.random.seed(random_state)

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        if encoding_gate.lower() == "rx":
            circuit.rx(param, qubit)
        elif encoding_gate.lower() == "ry":
            circuit.ry(param, qubit)
        elif encoding_gate.lower() == "rz":
            circuit.rz(param, qubit)
        else:
            raise ValueError(f"Unsupported encoding gate: {encoding_gate}")

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Measurement operators
    if measurement_bases is None:
        measurement_bases = ["Z"] * num_qubits
    observables: List[PauliSumOp] = []
    for i, base in enumerate(measurement_bases):
        pauli_str = "".join(base if j == i else "I" for j in range(num_qubits))
        observables.append(PauliSumOp.from_list([(pauli_str, 1.0)]))

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
