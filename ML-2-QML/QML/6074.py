"""Hybrid variational estimator with optional backend selection, multi‑shot sampling,
and noise‑aware expectation computation for quantum circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.backend import Backend
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterised quantum circuit to evaluate.
    backend : Backend | None, optional
        Backend to use for simulation.  If ``None`` an ``AerSimulator`` is created.
    shots : int | None, optional
        Number of shots to use.  If ``None`` (default) the state‑vector
        expectation value is returned deterministically.
    noise_model : NoiseModel | None, optional
        Optional noise model to attach to the simulator.
    seed : int | None, optional
        Random seed for the noise generator when using shots.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Backend | None = None,
        shots: Optional[int] = None,
        noise_model: NoiseModel | None = None,
        seed: Optional[int] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        if noise_model is not None:
            self.backend.set_options(noise_model=noise_model)
        self.shots = shots
        self.seed = seed

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables to evaluate.  Each operator is expected to be compatible
            with ``Statevector.expectation_value``.
        parameter_sets : sequence of parameter vectors
            Each vector is a 1‑D sequence of floats.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each containing the
            expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(self.seed)

        for values in parameter_sets:
            bound_circuit = self._bind(values)

            if self.shots is None:
                # Deterministic state‑vector expectation
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Use simulator with shots and add Gaussian noise
                job = execute(bound_circuit, backend=self.backend, shots=self.shots)
                result = job.result()
                state_dict = result.get_statevector(bound_circuit)
                state = Statevector(state_dict)
                row = [state.expectation_value(obs) for obs in observables]
                # Inject shot noise
                row = [complex(rng.normal(val.real, 1 / np.sqrt(self.shots)),
                               rng.normal(val.imag, 1 / np.sqrt(self.shots))) for val in row]

            results.append(row)

        return results

__all__ = ["FastBaseEstimator"]
