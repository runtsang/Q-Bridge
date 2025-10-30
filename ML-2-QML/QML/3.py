"""Core circuit factory for the incremental data‑uploading classifier.

The quantum circuit accepts a user‑defined measurement basis and an
encoding type.  The returned `circuit`, `encoding`, `weights` and
`observables` can be fed into a variational optimizer or a
quantum‑classical hybrid training loop.

The function signature matches the classical counterpart so that the two
models can be interchanged seamlessly.
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    measurement_basis: str = "Z",
    encoding_type: str = "rx",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational
    parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of repetitions of the variational layer.
    measurement_basis : str, default "Z"
        Basis used for the measurement observables.  Must be one of
        {"X", "Y", "Z"}.
    encoding_type : str, default "rx"
        Type of single‑qubit rotation used for the feature encoding.
        Options are {"rx", "ry", "rz"}.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : List[Parameter]
        List of encoding parameters.
    weights : List[Parameter]
        List of variational parameters.
    observables : List[SparsePauliOp]
        Measurement observables in the chosen basis.
    """
    if measurement_basis.upper() not in {"X", "Y", "Z"}:
        raise ValueError("measurement_basis must be one of 'X', 'Y', or 'Z'")
    if encoding_type.lower() not in {"rx", "ry", "rz"}:
        raise ValueError("encoding_type must be one of 'rx', 'ry', or 'rz'")

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode features
    for param, qubit in zip(encoding, range(num_qubits)):
        if encoding_type.lower() == "rx":
            circuit.rx(param, qubit)
        elif encoding_type.lower() == "ry":
            circuit.ry(param, qubit)
        else:
            circuit.rz(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entanglement: CZ between neighbours (cyclic)
        for qubit in range(num_qubits):
            circuit.cz(qubit, (qubit + 1) % num_qubits)

    # Observables
    pauli_dict = {"X": "X", "Y": "Y", "Z": "Z"}
    basis = pauli_dict[measurement_basis.upper()]
    observables = [
        SparsePauliOp(basis * i + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
