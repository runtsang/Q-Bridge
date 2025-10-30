"""
Hybrid estimator for Qiskit circuits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, List, Sequence

from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class FastHybridEstimator:
    """
    Evaluate expectation values for a parametrized Qiskit circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The variational circuit to be evaluated.

    Notes
    -----
    Deterministic state‑vector evaluation is used when *shots* is None.
    For finite‑shot estimates, the Aer qasm simulator is employed.
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
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of BaseOperator instances.
        parameter_sets
            Sequence of sequences of parameter values.
        shots
            If provided, perform finite‑shot simulation with the given shot count.
        seed
            Random seed for the simulator.

        Returns
        -------
        List[List[complex]]
            Rows of expectation values for each parameter set.
        """
        obs = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # Deterministic evaluation via Statevector
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(observable) for observable in obs]
                results.append(row)
        else:
            # Finite‑shot evaluation using Aer
            backend = Aer.get_backend("qasm_simulator")
            shots = int(shots)
            for values in parameter_sets:
                bound_circuit = self._bind(values)
                bound_circuit.save_statevector()
                job = backend.run(
                    bound_circuit, shots=shots, seed_simulator=seed
                )
                result = job.result()
                state = result.get_statevector(bound_circuit)
                row = [state.expectation_value(observable) for observable in obs]
                results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
