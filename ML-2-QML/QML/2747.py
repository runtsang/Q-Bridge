"""Quantum circuit builder for the hybrid classifier.

The circuit uses a basis encoding of the `feature_dim` classical features and a
layered ansatz of depth `depth`.  Observables are Pauli‑Z on each qubit,
matching the classical output indices.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    feature_dim: int = 4,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a quantum circuit that mirrors the classical metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (should match `feature_dim`).
    depth : int
        Depth of the variational ansatz.
    feature_dim : int
        Size of the feature vector encoded into qubits.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : Iterable
        Parameter names for the data encoding.
    weights : Iterable
        Parameter names for the variational weights.
    observables : List[SparsePauliOp]
        Pauli‑Z observables on each qubit.
    """
    # Data encoding: RX gates parameterized by the feature vector
    data_params = ParameterVector("x", length=feature_dim)
    weight_params = ParameterVector("theta", length=num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Basis encoding
    for qubit, param in enumerate(data_params):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weight_params[idx], qubit)
            idx += 1
        # Entanglement – CZ chain
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(data_params), list(weight_params), observables

__all__ = ["build_classifier_circuit"]
