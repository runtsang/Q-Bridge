"""Hybrid estimator for quantum circuits and quanvolution hybrids.

The estimator evaluates a :class:`qiskit.circuit.QuantumCircuit`, optionally
adding Gaussian shot noise to emulate finite sampling.  It also provides a
helper that constructs a variational quanvolution circuit by embedding a
2×2 patch subcircuit across a 28×28 image.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator


class FastBaseEstimator:
    """Evaluate quantum circuits with optional shot‑noise emulation."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Bind a parameter vector to the circuit."""
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        If *shots* is provided, Gaussian noise with variance 1/shots is added
        to each expectation value to emulate finite‑shot sampling.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    float(rng.normal(val.real, max(1e-6, 1 / shots))),
                    val.imag,
                )
                for val in row
            ]
            noisy.append(noisy_row)

        return noisy

    @staticmethod
    def build_quanvolution_circuit() -> QuantumCircuit:
        """Return a variational quanvolution circuit for a 28×28 image.

        The circuit applies a 2×2 patch subcircuit across the image and
        entangles the four qubits with a small random layer.
        """
        n_wires = 4
        base = QuantumCircuit(n_wires)

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Encode the 2×2 patch with Ry gates
                for i in range(n_wires):
                    base.ry(Parameter(f"p_{r}_{c}_{i}"), i)
                # Simple entanglement pattern
                base.cx(0, 1)
                base.cx(2, 3)
                base.cz(0, 2)
                base.cz(1, 3)

        return base
