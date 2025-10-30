"""Quantum‑enhanced estimator with gradient and shot‑noise simulation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
from qiskit.opflow import PauliOp, SummedOp
from qiskit.transpiler import transpile
from qiskit.providers.aer import AerSimulator


class FastBaseEstimator:
    """
    Evaluate expectation values of parametrised circuits with optional
    shot‑based measurement simulation and parameter‑shift gradient estimation.
    """

    def __init__(self, circuit: QuantumCircuit, backend: Optional[AerSimulator] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()

    # --------------------------------------------------------------------- #
    # Core evaluation
    # --------------------------------------------------------------------- #

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Pauli | PauliOp | SummedOp | BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        transpile_options: Optional[dict] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Qiskit operators (Pauli strings or summed operators).
        parameter_sets:
            Sequence of parameter sequences matching circuit parameters.
        shots:
            If provided, simulate measurements with `shots` repetitions.
        seed:
            Seed for the AerSimulator.
        transpile_options:
            Dictionary of options passed to `qiskit.transpiler.transpile`.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        circ = self._circuit
        if transpile_options:
            circ = transpile(circ, backend=self.backend, **transpile_options)

        for values in parameter_sets:
            bound_circ = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = self.backend.run(bound_circ, shots=shots, seed_simulator=seed)
                result = job.result()
                counts = result.get_counts(bound_circ)
                # Estimate expectation via measurement statistics
                row = [self._classical_expectation(counts, obs) for obs in observables]
            results.append(row)
        return results

    # --------------------------------------------------------------------- #
    # Gradient estimation
    # --------------------------------------------------------------------- #

    def gradient_via_parameter_shift(
        self,
        observables: Iterable[Pauli | PauliOp | SummedOp | BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute gradients of expectation values w.r.t. circuit parameters
        using the parameter‑shift rule.

        Returns a list of gradients matching the shape of `evaluate` output.
        """
        grads: List[List[float]] = []

        for values in parameter_sets:
            grad_row: List[float] = []
            for i, param in enumerate(self._parameters):
                # Shift parameters + and – by `shift`
                plus = list(values)
                minus = list(values)
                plus[i] += shift
                minus[i] -= shift
                # Evaluate expectation for shifted parameters
                exp_plus = self.evaluate(observables, [plus], shots=shots, seed=seed)[0]
                exp_minus = self.evaluate(observables, [minus], shots=shots, seed=seed)[0]
                # Parameter‑shift derivative
                grad = [(p_plus - p_minus) / 2 for p_plus, p_minus in zip(exp_plus, exp_minus)]
                grad_row.append(grad)
            grads.append(grad_row)
        return grads

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def _classical_expectation(counts: dict[str, int], observable: Pauli | PauliOp | SummedOp | BaseOperator) -> complex:
        """
        Estimate expectation value from measurement counts for a single Pauli operator.
        """
        if isinstance(observable, Pauli):
            pauli = observable
        elif isinstance(observable, PauliOp):
            pauli = observable.pauli
        else:
            raise TypeError("Only single Pauli operators are supported for measurement estimation.")

        exp_val = 0.0
        for bitstring, count in counts.items():
            # Compute eigenvalue (+1/-1) for the Pauli string
            eigen = 1
            for idx, pauli_char in enumerate(pauli.to_label()):
                if pauli_char!= 'I':
                    qubit = len(bitstring) - 1 - idx
                    bit = int(bitstring[qubit])
                    if pauli_char == 'Z':
                        eigen *= 1 if bit == 0 else -1
                    elif pauli_char == 'X':
                        eigen *= 1 if bit == 0 else -1  # Simplified; for real circuits X would need basis change
                    elif pauli_char == 'Y':
                        eigen *= 1 if bit == 0 else -1
            exp_val += eigen * count
        return exp_val / sum(counts.values())

__all__ = ["FastBaseEstimator"]
