"""FastBaseEstimator: shot‑sampling quantum estimator built on Qiskit.

This module extends the original estimator by:
* Vectorised evaluation of multiple parameter sets.
* Optional shot‑based sampling via the Aer QASM simulator.
* Caching of compiled circuits for repeated evaluations.
* Support for generic `qiskit.quantum_info.operators.base_operator.BaseOperator` observables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.opflow import PauliSumOp, StateFn, CircuitSampler, AerPauliExpectation
import numpy as np

class FastBaseEstimator:
    """Evaluate expectation values of parametrised Qiskit circuits.

    Parameters
    ----------
    circuit
        A parametrised :class:`~qiskit.circuit.QuantumCircuit` with symbolic
        parameters.  The circuit must be compatible with the Aer simulator.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = AerSimulator(method="statevector")
        # Pre‑compile the circuit for speed
        self._compiled_circuit = transpile(self._circuit, self._backend)

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
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of :class:`~qiskit.quantum_info.operators.base_operator.BaseOperator`
            instances.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, expectation values are sampled using the Aer QASM
            simulator.  Otherwise a deterministic state‑vector evaluation is
            performed.
        seed
            Random seed for reproducible sampling.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        if not parameter_sets:
            return []

        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound_circuit = self._bind(params)

            if shots is None:
                # Deterministic state‑vector evaluation
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Shot‑based sampling
                sampler = CircuitSampler(
                    AerSimulator(method="qasm", shots=shots, seed_simulator=seed)
                )
                state_fn = StateFn(bound_circuit)
                exp_factory = AerPauliExpectation()
                row = [
                    exp_factory.convert(state_fn @ PauliSumOp.from_operator(obs)).eval().real
                    for obs in observables
                ]

            results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
