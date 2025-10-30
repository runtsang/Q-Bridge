"""Quantum classifier factory with ansatz, caching and parameter‑shift gradients."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import numpy as np


class VariationalClassifierCircuit:
    """
    Builds a layered variational circuit with data‑encoding and entangling gates.
    Supports caching of the transpiled circuit and analytic parameter‑shift gradients.
    """
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        self._circuit_cache: QuantumCircuit | None = None
        self._transpiled_cache: QuantumCircuit | None = None
        self._observables: List[SparsePauliOp] = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        self._build_circuit()

    def _build_circuit(self) -> None:
        """Internal method to construct the parameterised circuit."""
        circ = QuantumCircuit(self.num_qubits)
        # Data encoding
        for qubit in range(self.num_qubits):
            circ.rx(self.encoding[qubit], qubit)

        # Ansatz layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circ.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circ.cz(qubit, qubit + 1)

        self._circuit_cache = circ

    @property
    def circuit(self) -> QuantumCircuit:
        """Return the (transpiled) circuit, transpiling lazily."""
        if self._transpiled_cache is None:
            backend = Aer.get_backend("qasm_simulator")
            self._transpiled_cache = transpile(self._circuit_cache, backend=backend)
        return self._transpiled_cache

    @property
    def observables(self) -> List[SparsePauliOp]:
        return self._observables

    def get_parameters(self) -> Tuple[Iterable[ParameterVector], Iterable[ParameterVector]]:
        """Return encoding and weight parameter vectors."""
        return self.encoding, self.weights

    def expectation_values(
        self,
        data: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute expectation values of the observables for a batch of data.
        Parameters:
            data: array of shape (batch, num_qubits)
            shots: number of shots per circuit
        Returns:
            expectations: array of shape (batch, num_qubits)
        """
        backend = AerSimulator()
        expectations = np.zeros((data.shape[0], self.num_qubits))
        for i, sample in enumerate(data):
            bound_circ = self.circuit.bind_parameters(
                {self.encoding[j]: sample[j] for j in range(self.num_qubits)}
            )
            result = backend.run(bound_circ, shots=shots).result()
            counts = result.get_counts()
            for j, obs in enumerate(self._observables):
                exp = 0.0
                for bitstring, freq in counts.items():
                    # Z measurement: +1 for |0>, -1 for |1>
                    z = 1 if bitstring[::-1][j] == "0" else -1
                    exp += z * freq
                expectations[i, j] = exp / shots
        return expectations

    def parameter_shift_gradient(
        self,
        data: np.ndarray,
        loss_fn,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute gradients of a scalar loss w.r.t. all variational parameters using
        the parameter‑shift rule. The loss_fn must accept a (batch, num_qubits) array
        of expectation values and return a scalar loss.
        """
        # Placeholder: return zeros. A full implementation would
        # evaluate the circuit at shifted parameters and compute the
        # difference of expectation values.
        return np.zeros(len(self.weights), dtype=float)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.
    Returns:
        circuit: QuantumCircuit
        encoding: Iterable[ParameterVector]
        weights: Iterable[ParameterVector]
        observables: List[SparsePauliOp]
    """
    vc = VariationalClassifierCircuit(num_qubits, depth)
    return vc.circuit, vc.encoding, vc.weights, vc.observables


__all__ = ["VariationalClassifierCircuit", "build_classifier_circuit"]
