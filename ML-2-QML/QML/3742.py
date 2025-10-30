"""Hybrid quantum classifier with a clipped data‑uploading ansatz.

The design borrows the encoding and variational layers from the QuantumClassifierModel seed
and adds a clipping mechanism inspired by the FraudDetection seed to keep parameters
within physically meaningful bounds.  The API matches the classical counterpart:
```
build_classifier_circuit(num_qubits, depth, clip=True)
```
returns the circuit, encoding, weight sizes and observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the symmetric bound."""
    return max(-bound, min(bound, value))


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    clip: bool = True,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    clip : bool, optional
        If True, clip all variational parameters to [-5, 5] after initialization,
        mirroring the clipping logic from the FraudDetection seed.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed quantum circuit.
    encoding : list[ParameterVector]
        Parameter vectors used for data encoding.
    weights : list[ParameterVector]
        Parameter vectors for the variational layers.
    observables : list[SparsePauliOp]
        Pauli-Z observables on each qubit, used as measurement targets.
    """
    # Data‑encoding parameters
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode data with Rx gates
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    param_idx = 0
    for _ in range(depth):
        # Rotate each qubit around Y
        for qubit in range(num_qubits):
            theta = weights[param_idx]
            if clip:
                theta = _clip(theta, 5.0)
            circuit.ry(theta, qubit)
            param_idx += 1

        # Entangle neighboring qubits with CZ
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
