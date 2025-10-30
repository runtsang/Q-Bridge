"""Quantum fast estimator with flexible backend and shot‑noise support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as _Iterable, List, Optional, Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import Aer, execute
from qiskit.transpiler import PassManager
from qiskit.providers.aer.noise import NoiseModel


ScalarObservable = Callable[[Statevector], complex]
Preprocess = Callable[[Sequence[float]], Sequence[float]]


def _ensure_batch(values: Sequence[float]) -> Sequence[Sequence[float]]:
    """Wrap a single parameter vector into a batch."""
    return [values] if isinstance(values[0], (int, float)) else values


class FastBaseEstimator:
    """Evaluate expectation values for a parametrized circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The base circuit with symbolic parameters.
    observables : Iterable[BaseOperator]
        Operators whose expectations are evaluated.
    backend : Optional[Callable], default None
        A qiskit provider backend (e.g., Aer.get_backend(\"statevector_simulator\")).  If ``None``, a state‑vector simulator is used.
    noise_model : Optional[NoiseModel], default None
        A noise model to attach to the backend.
    transpile_options : Optional[PassManager], default None
        Pass manager for additional transpilation passes.
    preprocess : Optional[Preprocess], default None
        A callable applied to raw parameter vectors before binding to the circuit.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: _Iterable[BaseOperator],
        *,
        backend: Optional[Callable] = None,
        noise_model: Optional[NoiseModel] = None,
        transpile_options: Optional[PassManager] = None,
        preprocess: Preprocess | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._observables = list(observables)
        self._backend = backend or Aer.get_backend("statevector_simulator")
        self._noise_model = noise_model
        self._transpile_options = transpile_options
        self._preprocess = preprocess

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        parameter_sets
            Sequence of parameter vectors.
        shots
            Number of shots for a measurement‑based backend.  If ``None`` the
            state‑vector simulator is used and the results are exact.
        """
        results: List[List[complex]] = []

        if shots is None:
            # Exact state‑vector evaluation
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in self._observables]
                results.append(row)
        else:
            # Measurement‑based evaluation with optional noise
            for values in parameter_sets:
                bound = self._bind(values)
                job = execute(
                    bound,
                    backend=self._backend,
                    shots=shots,
                    noise_model=self._noise_model,
                    transpile_options=self._transpile_options,
                )
                result = job.result()
                # Compute expectation values from measurement counts
                row = []
                for obs in self._observables:
                    exp = result.get_expectation_value(obs, bound)
                    row.append(exp)
                results.append(row)

        return results


class FastEstimator(FastBaseEstimator):
    """Same as :class:`FastBaseEstimator` but adds optional Gaussian shot noise.

    Parameters
    ----------
    shots : Optional[int], default None
        Number of shots to simulate.  When ``None`` the estimation is exact.
    seed : Optional[int], default None
        Seed for random number generator used in shot‑noise simulation.
    """

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(parameter_sets, shots=shots)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(complex(v.real, v.imag), 1 / shots) for v in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
