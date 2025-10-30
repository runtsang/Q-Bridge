"""Quantum estimator that evaluates expectation values for a parameterised circuit.

Features
--------
- Batch evaluation of many parameter sets.
- Optional shot noise via sampling.
- Compatible with Qiskit circuits and can be extended to TorchQuantum modules.
"""

import numpy as np
from typing import Iterable, Sequence, List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Optional TorchQuantum support
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    import torch
except ImportError:
    tq = None  # TorchQuantum not available


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised quantum circuit."""

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
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Qiskit operators whose expectation values are required.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors for the circuit.
        shots : int, optional
            Number of measurement shots to simulate sampling noise.
        seed : int, optional
            Random seed for reproducible shot noise.

        Returns
        -------
        List[List[complex]]
            Nested list with shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed)

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(op) for op in observables]

            if shots is not None:
                # Add Gaussian sampling noise around true expectation
                row = [
                    complex(
                        rng.normal(float(val.real), 1 / shots),
                        rng.normal(float(val.imag), 1 / shots),
                    )
                    for val in row
                ]

            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
