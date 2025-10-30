"""Quantum circuit that mirrors the hybrid classical classifier.

The circuit encodes each classical feature into a qubit via ry gates,
then applies a depth‑wise variational ansatz with entangling CZ gates.
Observables are Z measurements on each qubit, matching the classical
output dimension.  The function signature and returned metadata
(`encoding`, `weights`, `observables`) match the classical side,
facilitating joint training or transfer learning.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit,
                                                                   Iterable[ParameterVector],
                                                                   Iterable[ParameterVector],
                                                                   List[SparsePauliOp]]:
    """
    Construct a quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, one per classical feature.
    depth : int
        Depth of the variational ansatz.

    Returns
    -------
    circuit : QuantumCircuit
        The full variational circuit.
    encoding : Iterable[ParameterVector]
        Parameter vector for feature encoding.
    weights : Iterable[ParameterVector]
        Parameter vector for variational weights.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode classical features using ry gates
    for qubit, param in enumerate(encoding):
        circuit.ry(param, qubit)

    # Variational layers
    for layer in range(depth):
        for qubit, param in enumerate(weights[layer * num_qubits : (layer + 1) * num_qubits]):
            circuit.ry(param, qubit)
        # Entangle neighbouring qubits with CZ
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp(f"I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, encoding, weights, observables

__all__ = ["build_classifier_circuit"]
