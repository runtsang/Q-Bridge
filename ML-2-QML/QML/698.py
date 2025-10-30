"""FastBaseEstimator for quantum circuits with Aer and Pauli‑sum support."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliSumOp, PauliOp, StateFn, AerPauliExpectation, CircuitStateFn, ExpectationFactory
from qiskit.quantum_info import Statevector
from collections.abc import Iterable, Sequence
from typing import List, Union, Optional

# --------------------------------------------------------------------------- #
# Estimator classes
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Evaluate expectation values of Pauli‑sum observables for a parametrised circuit.
    Supports state‑vector simulation and shot‑noise sampling via Aer.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[AerSimulator] = None,
        shots: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterised circuit to evaluate.
        backend : AerSimulator, optional
            Aer backend; defaults to AerSimulator(statevector=True).
        shots : int, optional
            Number of shots for sampling; if None, uses state‑vector expectation.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        self.shots = shots
        if shots is None:
            self.backend = AerSimulator(method="statevector")
        self._cache: dict[tuple[float,...], Statevector] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Union[PauliSumOp, PauliOp, StateFn]],
        parameter_sets: Union[Sequence[Sequence[float]], np.ndarray],
    ) -> List[List[complex]] | np.ndarray:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable of PauliSumOp / PauliOp / StateFn
            Observables to evaluate.
        parameter_sets : Sequence[Sequence[float]] or np.ndarray
            Batch of parameter values.

        Returns
        -------
        List[List[complex]] or np.ndarray
            Expectation values for each parameter set.
        """
        observables = list(observables)
        if not observables:
            raise ValueError("At least one observable must be provided.")

        param_array = np.asarray(parameter_sets, dtype=np.float64)
        if param_array.ndim == 1:
            param_array = param_array.reshape(1, -1)

        results: List[List[complex]] = []

        for values in param_array:
            key = tuple(values)
            if key in self._cache:
                state = self._cache[key]
            else:
                bound_circ = self._bind(values)
                if self.shots is None:
                    state = Statevector.from_instruction(bound_circ)
                else:
                    # Use Aer to sample shots
                    job = self.backend.run(bound_circ, shots=self.shots)
                    counts = job.result().get_counts()
                    # Convert counts to statevector approximation
                    state = Statevector.from_counts(counts)
                self._cache[key] = state

            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return np.asarray(results, dtype=np.complex128)
        

__all__ = ["FastBaseEstimator"]
