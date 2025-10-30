"""FastBaseEstimator for quantum circuits with optional sampling."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrized quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized circuit whose parameters will be bound per eval call.

    Notes
    -----
    * Supports both analytical expectation values (via Statevector) and
      sampling (via Aer qasm_simulator) when `shots` is specified.
    * Observables must be `BaseOperator` instances (e.g., Pauli operators).
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
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            List of observables to evaluate.
        parameter_sets : Sequence[Sequence[float]]
            Parameter vectors to bind to the circuit.
        shots : int, optional
            If provided, perform sampling using Aer qasm_simulator.
        seed : int, optional
            Random seed for the simulator.

        Returns
        -------
        List[List[complex]]
            Nested list: outer over parameter sets, inner over observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # Analytical expectation values via Statevector
            for params in parameter_sets:
                bound = self._bind(params)
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        else:
            # Sampling via Aer qasm_simulator
            backend = Aer.get_backend("qasm_simulator")
            for params in parameter_sets:
                bound = self._bind(params)
                job = execute(
                    bound,
                    backend=backend,
                    shots=shots,
                    seed_simulator=seed,
                )
                result = job.result()
                counts = result.get_counts(bound)
                # Convert counts to expectation values
                row = []
                for obs in observables:
                    exp_val = 0.0
                    for bitstring, freq in counts.items():
                        # Evaluate eigenvalue of Pauli string on bitstring
                        eig = 1
                        for i, qubit in enumerate(reversed(bitstring)):
                            if obs.coeffs[0].data[i] == 1:  # Z operator
                                eig *= -1 if qubit == '1' else 1
                        exp_val += eig * freq / shots
                    row.append(complex(exp_val))
                results.append(row)

        return results


__all__ = ["FastBaseEstimator"]
