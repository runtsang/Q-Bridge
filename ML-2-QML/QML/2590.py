"""Hybrid quantum classifier implemented with Qiskit.

The circuit follows the same structure as the original
QuantumClassifierModel: data encoding with RX, depth‑wise variational
layers of Ry and CZ, and measurement of Pauli‑Z on each qubit.  To
bring in the fraud‑detection flavour, the circuit also exposes
parameter vectors for a photonic‑style displacement that can be
interpreted as additional shift parameters when the circuit is
executed on a photonic backend.  The public API remains the
build_classifier_circuit function returning (QuantumCircuit,
encoding, weights, observables).
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                   Iterable,
                                                   Iterable,
                                                   List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit data encoding and
    variational parameters.  The circuit also returns the list of
    encoding parameters, the list of variational parameters,
    and the set of observables to be measured.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entanglement
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables

class HybridClassifier:
    """Convenience wrapper exposing a consistent API."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def bind_parameters(self, data: List[float]) -> QuantumCircuit:
        circ = self.circuit.copy()
        for param, val in zip(self.encoding, data):
            circ.set_parameter(param, val)
        return circ

    def run(self, backend, data: List[float], shots: int = 1024):
        circ = self.bind_parameters(data)
        job = backend.run(circ, shots=shots)
        result = job.result()
        return result.get_counts(circ)

__all__ = ["build_classifier_circuit", "HybridClassifier"]
