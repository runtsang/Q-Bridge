"""Adaptive quantum estimator with zero‑noise extrapolation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import transpile
from qiskit.quantum_info.operators.base_operator import BaseOperator


class AdaptiveEstimator:
    """
    Evaluate expectation values of observables on a parametrised circuit with optional
    zero‑noise extrapolation.

    Parameters
    ----------
    circuit : QuantumCircuit
        The base circuit containing parameterised gates.
    noise_model : NoiseModel | None, optional
        Noise model used when simulating on AerSimulator.  If ``None`` the statevector
        simulator is used.
    backend : str | None, optional
        Name of the Aer simulator backend to use.  Defaults to ``"aer_simulator"``.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        noise_model: Optional[NoiseModel] = None,
        backend: str = "aer_simulator",
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.noise_model = noise_model
        self.backend_name = backend

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _simulate(
        self,
        param_values: Sequence[float],
        *,
        use_noise: bool = True,
    ) -> Statevector:
        bound = self._bind(param_values)

        if self.noise_model is None or not use_noise:
            return Statevector.from_instruction(bound)

        simulator = AerSimulator(noise_model=self.noise_model, backend_name=self.backend_name)
        transpiled = transpile(bound, simulator)
        result = simulator.run(transpiled).result()
        return Statevector(result.get_statevector(bound))

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        extrapolate: bool = True,
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        If ``extrapolate`` is ``True`` and a noise model is supplied, the
        expectation is linearly extrapolated to zero noise using results at
        noise levels 0× (ideal) and 1× (actual).

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Quantum operators whose expectation values are required.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        extrapolate : bool, optional
            Whether to apply zero‑noise extrapolation.

        Returns
        -------
        List[List[complex]]
            Nested list where each inner list contains the expectation values
            for a single parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            if self.noise_model is None or not extrapolate:
                state = self._simulate(values, use_noise=False if self.noise_model is None else True)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
                continue

            # Zero‑noise extrapolation with 0× (ideal) and 1× (actual)
            state_noisy = self._simulate(values, use_noise=True)
            state_ideal = self._simulate(values, use_noise=False)

            row = [
                2 * state_noisy.expectation_value(obs) - state_ideal.expectation_value(obs)
                for obs in observables
            ]
            results.append(row)

        return results


__all__ = ["AdaptiveEstimator"]
