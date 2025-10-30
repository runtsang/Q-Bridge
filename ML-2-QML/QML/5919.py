from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import BaseOperator, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """Fast expectation value evaluator for a parametrised circuit.

    The class is inspired by the original FastBaseEstimator but adds
    optional Gaussian shot noise and support for multiple observables.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of operators to evaluate.
        parameter_sets
            Sequence of parameter vectors that bind to the circuit.
        shots
            If provided, Gaussian noise with std ``1/√shots`` is added to each
            expectation value to emulate measurement shot noise.
        seed
            Random seed for reproducible noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy.append(
                    [
                        rng.normal(complex(v).real, max(1e-6, 1 / shots))
                        + 1j * rng.normal(complex(v).imag, max(1e-6, 1 / shots))
                        for v in row
                    ]
                )
            return noisy

        return results


class EstimatorQNN:
    """Hybrid quantum estimator that uses a parametrised circuit and a
    FastBaseEstimator for efficient batched evaluation.

    The circuit is constructed from user supplied parameters.  The
    ``evaluate`` method delegates to FastBaseEstimator and can add
    shot noise.  The design mirrors the original EstimatorQNN but
    incorporates the FastBaseEstimator logic for speed.

    Parameters
    ----------
    circuit
        Parameterised quantum circuit.
    observables
        Sequence of operators to evaluate.
    input_params
        Parameters that encode the classical input.
    weight_params
        Parameters that represent trainable circuit weights.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: Sequence[BaseOperator],
        input_params: Sequence[Parameter],
        weight_params: Sequence[Parameter],
    ) -> None:
        self.circuit = circuit
        self.observables = list(observables)
        self.input_params = list(input_params)
        self.weight_params = list(weight_params)
        self.estimator = FastBaseEstimator(circuit)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate the circuit for a batch of parameter sets.

        The input parameters are expected to be the first ``len(input_params)``
        values of each set, followed by the weight parameters.  The method
        forwards the full set to the underlying FastBaseEstimator.
        """
        return self.estimator.evaluate(
            self.observables,
            parameter_sets,
            shots=shots,
            seed=seed,
        )


__all__ = ["EstimatorQNN", "FastBaseEstimator"]
