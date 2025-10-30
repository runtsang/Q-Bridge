"""Quantum circuit builder with data re‑uploading and multi‑qubit observables.

The `ClassifierCircuitBuilder` class encapsulates a variational ansatz that
iteratively re‑uploads the data, applies parameterised rotations and a
custom entanglement pattern.  The implementation extends the seed by
adding a second entanglement layer and a multi‑qubit Z‑Z observable
for richer measurement statistics.  The public API matches the seed
for seamless substitution.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class ClassifierCircuitBuilder:
    """Factory for a data‑re‑uploading variational classifier."""
    
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with data re‑uploading.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the circuit.
        depth:
            Number of data‑re‑uploading layers.

        Returns
        -------
        circuit:
            QuantumCircuit object.
        encoding:
            ParameterVector of data‑encoding parameters.
        weights:
            ParameterVector of variational parameters.
        observables:
            List of SparsePauliOp objects to measure the Z‑basis on each qubit
            plus a global ZZ observable for parity information.
        """
        # Data encoding parameters
        encoding = ParameterVector("x", num_qubits)

        # Variational parameters: two rotation angles per qubit per layer
        weights = ParameterVector("theta", num_qubits * depth * 2)

        circuit = QuantumCircuit(num_qubits)

        # Initial data encoding
        for idx, qubit in enumerate(range(num_qubits)):
            circuit.rx(encoding[idx], qubit)

        weight_idx = 0
        for layer in range(depth):
            # Re‑upload data (optional)
            if layer > 0:
                for idx, qubit in enumerate(range(num_qubits)):
                    circuit.rx(encoding[idx], qubit)

            # Parameterised rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[weight_idx], qubit)
                weight_idx += 1
                circuit.rz(weights[weight_idx], qubit)
                weight_idx += 1

            # Entanglement: alternating CZ and CX to increase connectivity
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        # Measurement observables
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        # Global ZZ observable for parity
        observables.append(SparsePauliOp("Z" * num_qubits))

        return circuit, encoding, weights, observables


# Back‑compatibility shim
build_classifier_circuit = ClassifierCircuitBuilder.build_classifier_circuit

__all__ = ["ClassifierCircuitBuilder", "build_classifier_circuit"]
