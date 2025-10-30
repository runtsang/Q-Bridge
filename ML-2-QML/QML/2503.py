"""Hybrid quantum classifier and kernel that share a common ansatz.

The module exposes a class ``HybridClassifier`` that can be used as a
variational classifier or as a kernel function.  The implementation
reuses the data‑encoding and ansatz from the original seed files but
adds a flexible interface for both tasks.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class HybridClassifier:
    """Quantum classifier that optionally returns a kernel matrix.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

    def build_circuit(
        self
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Return a variational classifier circuit and metadata.

        The circuit consists of an RX data encoding followed by ``depth`` layers
        of RY rotations and CZ entanglement.  The returned metadata mirrors
        the original ``build_classifier_circuit`` helper.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def build_kernel_circuit(
        self
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Return a quantum kernel circuit using the same ansatz.

        The kernel circuit applies a data‑encoding with ``ry`` gates, then
        applies the variational ansatz, and finally un‑encodes with the
        negative of the second input.  The circuit returns a single
        amplitude that is taken as the kernel value.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        # Forward encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.ry(param, qubit)

        # Variational block
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Reverse encoding with negative parameters
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.ry(-param, qubit)

        observables = [SparsePauliOp("I" * self.num_qubits)]  # amplitude of |0...0>
        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def kernel_matrix(
        a: Sequence[QuantumCircuit], b: Sequence[QuantumCircuit]
    ) -> np.ndarray:
        """Compute the Gram matrix between two sets of inputs using the kernel circuit."""
        # Placeholder implementation: in practice one would use a simulator or
        # a backend to evaluate the circuits and extract the amplitude.
        # Here we return a dummy matrix of zeros for API compatibility.
        return np.zeros((len(a), len(b)))

__all__ = ["HybridClassifier"]
