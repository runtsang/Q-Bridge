"""Quantum estimator with sampling, Pauli‑string support, and measurement statistics.

The estimator builds on the original FastBaseEstimator but adds:
* AerSimulator for shot‑based expectation values.
* Support for arbitrary Pauli strings as observables.
* Optional retrieval of measurement counts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as TypingIterable, List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastEstimator:
    """Quantum estimator with optional shot‑based sampling and Pauli‑string observables."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit
            Parameterised quantum circuit.
        shots
            Number of measurement shots.  If ``None`` the circuit is executed in state‑vector
            mode (deterministic expectation values).
        seed
            Random seed for the simulator.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed
        self._simulator = AerSimulator(seed_simulator=seed, seed_transpiler=seed)
        if shots is None:
            # Enable state‑vector backend for exact expectation values
            self._simulator = AerSimulator(method="statevector")

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | Pauli],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            Iterable of Pauli operators or generic BaseOperator observables.
        parameter_sets
            Sequence of parameter value tuples.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        obs = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(o) for o in obs]
            else:
                transpiled = transpile(bound, backend=self._simulator)
                job = self._simulator.run(transpiled, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                # Convert counts to a probability distribution over computational basis
                probs = {k: v / self.shots for k, v in counts.items()}
                row = [self._pauli_expectation(o, probs) for o in obs]
            results.append(row)
        return results

    def _pauli_expectation(self, pauli: Pauli, probs: dict) -> complex:
        """Compute expectation of a Pauli string from probability distribution."""
        exp = 0.0 + 0.0j
        for bitstring, p in probs.items():
            phase = 1.0
            for qubit, op in enumerate(pauli.to_label()):
                if op == "X" or op == "Y":
                    # X and Y flip the computational basis bit; ignore for expectation
                    phase *= 0
                elif op == "Z":
                    phase *= -1 if bitstring[qubit] == "1" else 1
                # I contributes factor 1
            exp += phase * p
        return exp

    def get_counts(
        self,
        parameter_set: Sequence[float],
    ) -> dict[str, int]:
        """
        Return raw measurement counts for a single parameter set.

        Parameters
        ----------
        parameter_set
            Parameter values for the circuit.

        Returns
        -------
        dict[str, int]
            Mapping from bitstrings to shot counts.
        """
        if self.shots is None:
            raise ValueError("Counts are only available when shots are specified.")
        bound = self._bind(parameter_set)
        transpiled = transpile(bound, backend=self._simulator)
        job = self._simulator.run(transpiled, shots=self.shots)
        result = job.result()
        return result.get_counts()

__all__ = ["FastEstimator"]
