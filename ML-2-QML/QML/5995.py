"""Quantum classifier built with a variational circuit and data re‑uploading."""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_quantum_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    The circuit encodes input data with RX gates, followed by depth‑number of
    trainable rotation layers and entangling CZ gates.  The observables are
    single‑qubit Z operators, mirroring the output dimensionality of the
    classical classifier.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers with entanglement
    weight_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class QuantumHybridClassifier:
    """Wrapper around the variational circuit for easy integration.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the variational ansatz.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weights, self.observables = (
            build_quantum_classifier_circuit(num_qubits, depth)
        )

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_parameters(self) -> List[ParameterVector]:
        """Return the encoding and variational parameters."""
        return [self.encoding_params, self.weights]

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the measurement observables (Z on each qubit)."""
        return self.observables


__all__ = ["QuantumHybridClassifier", "build_quantum_classifier_circuit"]
