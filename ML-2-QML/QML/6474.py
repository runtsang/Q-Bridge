"""Hybrid estimator that evaluates quantum circuits with deterministic and sampling modes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastHybridEstimator:
    """Evaluate a parametrized quantum circuit for batches of parameters.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to be evaluated. All parameters must be declared.
    device : str | None, optional
        Backend name. ``None`` defaults to the local Aer simulator.
    shots : int | None, optional
        If provided, evaluation uses a qasm simulator with the given number of shots
        and returns expectation values estimated from measurement samples.
        If ``None`` (default), a statevector simulator is used for exact
        expectation values.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        device: str | None = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)
        self._shots = shots
        if shots is None:
            self._backend = Aer.get_backend(device or "statevector_simulator")
        else:
            self._backend = Aer.get_backend(device or "qasm_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Deterministic evaluation using statevector simulator.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators for which expectation values are computed.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the values to bind to the circuit.

        Returns
        -------
        List[List[complex]]
            Rows of expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_sampling(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Probabilistic evaluation using QASM simulator.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators for which expectation values are estimated from samples.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the values to bind to the circuit.

        Returns
        -------
        List[List[complex]]
            Rows of expectation values estimated from measurement samples.
        """
        if self._shots is None:
            raise RuntimeError("Shots must be defined for sampling mode.")
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            job = execute(bound, self._backend, shots=self._shots)
            counts = job.result().get_counts()
            state = Statevector.from_counts(counts)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["FastHybridEstimator"]
