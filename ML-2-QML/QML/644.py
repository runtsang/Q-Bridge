"""FastBaseEstimatorGen301: Variational circuit evaluator with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.result import Result


class FastBaseEstimatorGen301:
    """Evaluate a parametrized quantum circuit for many parameter sets.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit that contains named parameters. Must be a valid Qiskit
        circuit that can be executed on a statevector or simulator backend.
    backend : str, optional
        Backend name for shot‑based simulation. Use ``'statevector'`` for
        exact evaluation or ``'aer'`` for sampling with shots.
    """

    def __init__(self, circuit: QuantumCircuit, backend: str = "statevector") -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend_name = backend
        if backend == "statevector":
            self._simulator = Statevector
        else:
            self._simulator = AerSimulator()
            self._simulator.set_options(shots=1)  # placeholder; actual shots set in evaluate

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _diag_expectation(self, obs: Operator, counts: dict[str, int], shots: int) -> complex:
        """Compute expectation value from measurement counts for diagonal operators."""
        diag = obs.data.diagonal()
        exp = 0 + 0j
        for bitstring, freq in counts.items():
            idx = int(bitstring[::-1], 2)  # Qiskit uses little‑endian bitstrings
            exp += (freq / shots) * diag[idx]
        return exp

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of qiskit.quantum_info.Operator
            Observables whose expectation values are to be evaluated.
        parameter_sets : sequence of sequence of floats
            Parameter values to bind to the circuit.
        shots : int, optional
            If provided, the circuit is executed with the given number of shots
            on an Aer simulator. The result is the mean of the measurement
            outcomes weighted by the observable eigenvalues.
        seed : int, optional
            Random seed for the Aer simulator.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each row containing
            the expectation values for the supplied observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None or self.backend_name == "statevector":
            # Exact statevector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        if self.backend_name!= "aer":
            raise ValueError("Shot simulation requires an Aer backend.")

        self._simulator.set_options(shots=shots, seed_simulator=seed)
        for values in parameter_sets:
            bound = self._bind(values)
            job = self._simulator.run(bound)
            result: Result = job.result()
            counts = result.get_counts()
            row: List[complex] = []
            for obs in observables:
                exp_val = self._diag_expectation(obs, counts, shots)
                row.append(exp_val)
            results.append(row)
        return results


__all__ = ["FastBaseEstimatorGen301"]
