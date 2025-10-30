"""HybridClassifier – quantum counterpart mirroring the classical architecture.

The class builds a layered ansatz with data‑encoding and variational parameters
matching the depth parameter of the classical network.  It exposes an
`evaluate` method that returns expectation values of Pauli‑Z observables,
and supports optional shot‑noise emulation via Gaussian sampling.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np

class HybridClassifier:
    """Quantum circuit that mirrors the classical HybridClassifier."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_circuit()
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(self.encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.encoding) + len(self.weights):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.encoding, parameter_values[:len(self.encoding)]))
        mapping.update(dict(zip(self.weights, parameter_values[len(self.encoding):])))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables: Optional[Iterable[BaseOperator]]
            Operators to measure; defaults to the class’ Pauli‑Z set.
        parameter_sets: Optional[Sequence[Sequence[float]]]
            Iterable of parameter vectors (encoding + variational).
        shots: Optional[int]
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value to emulate shot noise.
        seed: Optional[int]
            Seed for the noise generator.
        """
        if observables is None:
            observables = self.observables
        if parameter_sets is None:
            raise ValueError("parameter_sets must be provided")
        observables = list(observables)

        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [float(rng.normal(complex(v).real, max(1e-6, 1 / shots))) for v in row]
                noisy.append(noisy_row)
            return noisy
        return results

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[HybridClassifier, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a quantum ansatz and metadata similar to the classical variant."""
    model = HybridClassifier(num_qubits, depth)
    encoding = model.encoding
    weights = model.weights
    observables = model.observables
    return model, encoding, weights, observables
