"""FastBaseEstimator for Qiskit circuits with caching and shot simulation."""
import numpy as np
from qiskit import Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import QuantumCircuit
from typing import Iterable, List, Sequence, Tuple, Dict

class FastBaseEstimator:
    """Estimator for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized quantum circuit to evaluate.
    cache : bool, default=False
        Enable caching of statevectors for repeated parameter sets.
    """

    def __init__(self, circuit: QuantumCircuit, *, cache: bool = False) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._cache_enabled = cache
        self._cache: Dict[Tuple[float,...], Statevector] = {}

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _statevector(self, param_values: Sequence[float]) -> Statevector:
        key = tuple(param_values)
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        circuit = self._bind(param_values)
        sv = Statevector.from_instruction(circuit)
        if self._cache_enabled:
            self._cache[key] = sv
        return sv

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            List of operators to measure.
        parameter_sets : Sequence[Sequence[float]]
            Parameter configurations.
        shots : int, optional
            If provided, add Gaussian shot noise with standard deviation 1/sqrt(shots).
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        observables = list(observables) or [Statevector.from_instruction(self._circuit).to_operator()]
        rng = np.random.default_rng(seed)
        results: List[List[complex]] = []

        for params in parameter_sets:
            sv = self._statevector(params)
            row = [sv.expectation_value(obs) for obs in observables]
            if shots is not None:
                std = max(1e-6, 1 / np.sqrt(shots))
                row = [rng.normal(loc=val.real, scale=std) for val in row]
            results.append(row)
        return results

    def clear_cache(self) -> None:
        """Clear the cached statevectors."""
        self._cache.clear()


__all__ = ["FastBaseEstimator"]
