"""Quantum circuit factory for a hybrid classifier mirroring the classical helper."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered variational circuit that accepts classical features encoded
    in the initial rotation angles.  The circuit is compatible with the
    :class:`QuantumClassifierModel` defined in the classical module.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, typically equal to the auto‑encoder latent dimension.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled circuit.
    encoding : Iterable[ParameterVector]
        Parameters representing the classical feature embed (Rx angles).
    weights : Iterable[ParameterVector]
        Variational parameters of the ansatz.
    observables : List[SparsePauliOp]
        Measurement operators returning the two‑class logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Feature embedding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observable: one Z per qubit, summed to give two‑class logits
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, [encoding], [weights], observables

__all__ = ["build_classifier_circuit"]
