"""Hybrid quantum estimator that evaluates parameterised quantum circuits
with optional simulated shot noise."""
from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Union

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Import the original quantum base class
from.FastBaseEstimator import FastBaseEstimator

# Optional: import the QCNN quantum ansatz
# (This path assumes the QCNN.py file provides a `QCNN` function that returns an EstimatorQNN)
try:
    from.QCNN import QCNN as quantum_qcnn_factory
except Exception:  # pragma: no cover
    quantum_qcnn_factory = None


class HybridEstimator(FastBaseEstimator):
    """
    Evaluator for parameterised quantum circuits with optional shot noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.  Must have parameters that can be bound
        to the given parameter sets.

    Notes
    -----
    * The default evaluation uses the exact statevector expectation values.
    * If *shots* is provided, Gaussian noise with variance ``1/shots`` is added
      to each expectation value to emulate measurement statistics.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)
        # Preserve the parameter list for quick binding
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
        """
        Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Qiskit operators to evaluate.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of 1‑D parameter lists.
        shots : int, optional
            If provided, add Gaussian noise with variance ``1/shots``.
        seed : int, optional
            Seed for the noise generator.

        Returns
        -------
        List[List[complex]]
            A matrix of shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row: List[complex] = []
            for obs in observables:
                exp_val = state.expectation_value(obs)
                if shots is not None:
                    noise = rng.normal(0, np.sqrt(1.0 / shots))
                    exp_val = exp_val + noise
                row.append(exp_val)
            results.append(row)
        return results

    @staticmethod
    def create_qcnn() -> "HybridEstimator":
        """
        Convenience constructor that returns a HybridEstimator for the QCNN
        quantum ansatz defined in the QCNN.py module.

        Returns
        -------
        HybridEstimator
            An estimator wrapping the QCNN ansatz circuit.
        """
        if quantum_qcnn_factory is None:
            raise RuntimeError("QCNN quantum factory not available.")
        qnn = quantum_qcnn_factory()
        # qnn.circuit is the underlying QuantumCircuit
        return HybridEstimator(qnn.circuit)
