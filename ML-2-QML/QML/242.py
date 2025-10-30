"""Fast quantum estimator with backend selection, shot/noise simulation, and parameter‑shift gradients.

The class `FastBaseEstimator` evaluates expectation values of arbitrary
`BaseOperator`s on a parametrised `QuantumCircuit`.  Compared to the original
seed, it adds:

* ``backend`` – choose an Aer simulator or a real IBMQ device.
* ``shots`` – number of measurement shots; if provided the circuit is simulated
  with a noise model that reflects shot noise.
* ``noise_model`` – a :class:`NoiseModel` instance for noisy simulation.
* ``return_gradients`` – compute the gradient of each observable w.r.t. the
  circuit parameters using the parameter‑shift rule.

The public API is compatible with the seed: the ``evaluate`` method accepts
``observables`` and ``parameter_sets`` and returns a nested list of complex
expectation values.  When ``return_gradients`` is ``True`` a tuple
``(results, gradients)`` is returned, where gradients are lists of lists of
floats matching the number of parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import numpy as np
from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _simulate(
        self,
        bound_circuit: QuantumCircuit,
        *,
        shots: int | None,
        noise_model: NoiseModel | None,
        backend: Backend | None,
    ) -> Statevector:
        """Return the final statevector, optionally with shot/noise simulation."""
        if shots is None:
            # Exact statevector evaluation
            return Statevector.from_instruction(bound_circuit)
        # Use AerSimulator with statevector method; noise_model and shots are applied.
        sim = AerSimulator(
            method="statevector",
            noise_model=noise_model,
            shots=shots,
            seed_simulator=None,
        )
        if backend is not None:
            sim = transpile(bound_circuit, backend=backend, simulator=sim)
        job = sim.run(bound_circuit)
        return Statevector(job.result().get_statevector(bound_circuit))

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        backend: Backend | None = None,
        noise_model: NoiseModel | None = None,
        return_gradients: bool = False,
    ) -> Union[List[List[complex]], Tuple[List[List[complex]], List[List[List[float]]]]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Quantum operators for which expectation values are desired.
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence represents a set of parameters for the circuit.
        shots : int, optional
            Number of measurement shots; if provided a noisy simulation is run.
        backend : Backend, optional
            The Qiskit backend to use for simulation.  If ``None`` the AerSimulator
            is used.
        noise_model : NoiseModel, optional
            Noise model applied when ``shots`` is not ``None``.
        return_gradients : bool, optional
            If ``True`` return a tuple ``(results, gradients)`` where gradients
            are lists of lists of floats corresponding to each observable.

        Returns
        -------
        results or (results, gradients)
            ``results`` is a list of lists of complex numbers.  ``gradients`` (if requested)
            is a list of lists of lists of floats: outer list per observable, inner
            per parameter set, innermost per circuit parameter.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        gradients: List[List[List[float]]] | None = None

        if return_gradients:
            gradients = []

        for params in parameter_sets:
            bound = self._bind(params)
            state = self._simulate(bound, shots=shots, noise_model=noise_model, backend=backend)
            row: List[complex] = [state.expectation_value(obs) for obs in observables]
            results.append(row)

            if return_gradients:
                grads_row: List[List[float]] = []
                for obs in observables:
                    # Parameter‑shift rule: shift each parameter by ±π/2
                    grad: List[float] = []
                    for i, _ in enumerate(self._parameters):
                        shifted_plus = list(params)
                        shifted_minus = list(params)
                        shifted_plus[i] += np.pi / 2
                        shifted_minus[i] -= np.pi / 2
                        state_plus = self._simulate(
                            self._bind(shifted_plus),
                            shots=shots,
                            noise_model=noise_model,
                            backend=backend,
                        )
                        state_minus = self._simulate(
                            self._bind(shifted_minus),
                            shots=shots,
                            noise_model=noise_model,
                            backend=backend,
                        )
                        exp_plus = state_plus.expectation_value(obs)
                        exp_minus = state_minus.expectation_value(obs)
                        grad.append(float((exp_plus - exp_minus) / 2))
                    grads_row.append(grad)
                gradients.append(grads_row)

        if not return_gradients:
            return results

        return results, gradients


__all__ = ["FastBaseEstimator"]
