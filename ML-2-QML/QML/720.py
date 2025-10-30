"""Quantum circuit factory with enriched entanglement and measurement.

The builder now constructs a parameterised ansatz that alternates
RX‑encoding, RZ‑rotations and a layer of CZ gates.  The depth controls how
many such layers are stacked.  The function returns the circuit together
with lists of encoding parameters, variational parameters and a set of
observable Pauli‑Z operators, matching the classical API.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered variational classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the feature dimensionality.
    depth : int
        Number of variational layers (each contains a rotation and an entangling block).

    Returns
    -------
    circuit : QuantumCircuit
        Assembled variational circuit.
    encoding : Iterable[ParameterVector]
        Parameters for data encoding (one RX per qubit).
    weights : Iterable[ParameterVector]
        Variational parameters (RZ rotations).
    observables : List[SparsePauliOp]
        Pauli‑Z operators for each qubit, to be measured by the backend.
    """
    # Encode data with RX rotations
    encoding = ParameterVector("x", length=num_qubits)
    # Variational parameters: one RZ per qubit per layer
    weights = ParameterVector("theta", length=num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # RX encoding
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    # Variational layers
    weight_idx = 0
    for _ in range(depth):
        # RZ rotations
        for q in range(num_qubits):
            circuit.rz(weights[weight_idx], q)
            weight_idx += 1
        # Entangling CZ block (full‑chain)
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, encoding, weights, observables


# Optional helper to run expectation values on Aer simulator
def simulate_expectations(circuit: QuantumCircuit, observables: List[SparsePauliOp], shots: int = 1024) -> List[float]:
    """Return expectation values of the given observables using Aer."""
    simulator = AerSimulator(method="statevector")
    job = simulator.run(circuit)
    result = job.result()
    statevector = result.get_statevector(circuit)
    return [op.eval(statevector).real for op in observables]


__all__ = ["build_classifier_circuit", "simulate_expectations"]
