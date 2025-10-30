"""Hybrid estimator for quantum models with optional shot‑noise.

This module mirrors the classical version but operates on Qiskit
`QuantumCircuit` objects.  It exposes the same public API and helper
constructors, enabling side‑by‑side experiments with classical and
quantum models.

Key features
------------
- `evaluate` accepts `BaseOperator` observables and returns complex
  expectation values.
- Optional shot‑noise is simulated by adding Gaussian perturbations to
  the deterministic result.
- Static helpers `build_classifier_circuit`, `SamplerQNN` and `FCL`
  provide quantum equivalents of the classical components.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.sparse_pauli_op import SparsePauliOp

class FastHybridEstimator:
    """Unified estimator for quantum models."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Quantum evaluation ------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def _evaluate_quantum(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> list[list[complex]]:
        observables = list(observables)
        results: list[list[complex]] = []
        for values in parameter_sets:
            bound = self._bind_quantum(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        # Simulate shot noise by Gaussian perturbation
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: list[list[complex]] = []
            for row in results:
                noisy_row = [
                    complex(rng.normal(v.real, 1 / shots))
                    + 1j * rng.normal(v.imag, 1 / shots)
                    for v in row
                ]
                noisy.append(noisy_row)
            return noisy
        return results

    def _bind_quantum(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    # ------------------------------------------------------------------ #
    # Public API -------------------------------------------------------- #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        """Evaluate the quantum circuit for every parameter set."""
        return self._evaluate_quantum(
            observables, parameter_sets, shots or self.shots, seed or self.seed
        )

    # ------------------------------------------------------------------ #
    # Helper constructors ----------------------------------------------- #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[
        QuantumCircuit, Iterable, Iterable, list[BaseOperator]
    ]:
        """Return a simple layered ansatz with explicit encoding and variational parameters."""
        encoding = qiskit.circuit.ParameterVector("x", num_qubits)
        weights = qiskit.circuit.ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp.from_label("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def SamplerQNN() -> tuple[QuantumCircuit, Iterable, Iterable]:
        """Return a simple parameterized sampler circuit."""
        inputs = qiskit.circuit.ParameterVector("input", 2)
        weights = qiskit.circuit.ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        # Measurement for sampling
        qc.measure_all()
        return qc, list(inputs), list(weights)

    @staticmethod
    def FCL() -> tuple[QuantumCircuit, Iterable]:
        """Return a single‑qubit circuit that emulates a fully‑connected layer."""
        circuit = QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circuit.h(0)
        circuit.ry(theta, 0)
        circuit.measure_all()
        return circuit, [theta]

__all__ = ["FastHybridEstimator"]
