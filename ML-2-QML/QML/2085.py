from __future__ import annotations

from typing import Iterable, List, Sequence, Optional
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np


class FastHybridEstimator:
    """Quantum estimator that evaluates expectation values of parametrised circuits.

    The estimator can run the circuit on a state‑vector backend or on a shot‑based
    AerSimulator.  For shot‑based runs the expectation of PauliZ products is
    constructed from measurement counts.  The API mirrors the classical
    :class:`FastHybridEstimator` for easy hybrid workflows.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._simulator = AerSimulator(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _counts_expectation(
        self,
        counts: dict,
        observable: BaseOperator,
        param_values: Sequence[float],
    ) -> complex:
        """Compute expectation from measurement counts for PauliZ products."""
        if isinstance(observable, Pauli):
            pauli_str = observable.paulis
            total = sum(counts.values())
            exp_val = 0.0
            for bitstring, count in counts.items():
                val = 1
                for qubit, op in enumerate(pauli_str):
                    if op == "Z":
                        val *= 1 if bitstring[-qubit - 1] == "0" else -1
                exp_val += val * count
            return exp_val / total
        # Fallback to state‑vector expectation
        state = Statevector.from_instruction(self._bind(param_values))
        return state.expectation_value(observable)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Evaluate expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of :class:`~qiskit.quantum_info.operators.base_operator.BaseOperator`
            for which the expectation value is required.
        parameter_sets
            Sequence of parameter vectors.
        shots
            If provided, the circuit is executed on an AerSimulator with the
            specified number of shots.  For PauliZ observables the expectation
            value is reconstructed from the measurement counts.
        seed
            Random seed for the AerSimulator.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # State‑vector backend
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑based simulation
        shot_sim = AerSimulator()
        if seed is not None:
            shot_sim.set_options(seed_simulator=seed)
        for values in parameter_sets:
            bound = self._bind(values)
            bound.measure_all()
            job = shot_sim.run(bound, shots=shots)
            result = job.result()
            counts = result.get_counts()
            row = [
                self._counts_expectation(counts, obs, values) for obs in observables
            ]
            results.append(row)

        return results


__all__ = ["FastHybridEstimator"]
