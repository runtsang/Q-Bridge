"""Hybrid quantum classifier that emulates the classical feed‑forward
architecture using a parameterized ansatz and Pauli‑Z measurements.

The circuit consists of:

* An explicit feature encoding using RX gates.
* A depth‑controlled variational block with Ry rotations and CZ
  entanglement.
* Measurement of each qubit in the Z basis to produce binary outcomes
  that can be interpreted as logits.

The function returns the circuit, lists of encoding and variational
parameters, and a set of observables that match the classical
metadata.  This design allows direct comparison of parameter counts
and output dimensionality.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a quantum circuit that parallels the classical feed‑forward
    model defined in the ML side.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the dimensionality of the input feature.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The parameterized quantum circuit.
    encoding : Iterable[ParameterVector]
        Parameter vectors for the feature encoding.
    weights : Iterable[ParameterVector]
        Parameter vectors for the variational layers.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
    # Feature encoding
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode each feature into an RX rotation
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational block
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangle neighboring qubits with CZ
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], [weights], observables


class HybridQuantumCircuit:
    """
    Wrapper that exposes the circuit building routine and metadata in a
    class interface matching the classical `HybridClassifier`.  This
    facilitates joint training pipelines that alternate between
    classical and quantum updates.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_parameters(self) -> Tuple[Iterable[ParameterVector], Iterable[ParameterVector]]:
        return self.encoding, self.weights


__all__ = ["build_classifier_circuit", "HybridQuantumCircuit"]
