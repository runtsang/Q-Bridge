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
    Quantum estimator for parametrized circuits.

    Features
    --------
    * Expectation value evaluation with optional shot noise.
    * Gradient computation via parameter‑shift rule.
    * Backend selection (statevector or qasm simulator).
    * Support for vectorised observables.
    """

    def __init__(self, circuit: QuantumCircuit, backend: Optional[BaseOperator] = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.qasm_backend = Aer.get_backend("qasm_simulator")

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables: iterable of BaseOperator
            Operators for which the expectation value is required.
        parameter_sets: sequence of parameter sequences
            Each inner sequence defines a point in parameter space.
        shots: int, optional
            If provided, the expectation values are estimated from
            a finite number of shots using the QASM simulator.
        seed: int, optional
            Seed for the random number generator in shot simulation.

        Returns
        -------
        List[List[complex]]
            List of rows, each containing the expectation values
            for the corresponding parameter set.
        """
        results: List[List[complex]] = []

        for params in parameter_sets:
            circ = self._bind(params)

            if shots is None:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    circ,
                    backend=self.qasm_backend,
                    shots=shots,
                    seed_simulator=seed,
                )
                result = job.result()
                counts = result.get_counts(circ)
                probs = {k: v / shots for k, v in counts.items()}
                row = []
                for obs in observables:
                    exp = 0 + 0j
                    for bitstring, prob in probs.items():
                        # Convert bitstring to computational basis state
                        matrix = obs.data
                        idx = int(bitstring, 2)
                        exp += prob * matrix[idx, idx]
                    row.append(exp)
            results.append(row)

        return results

    def gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
    ) -> List[List[complex]]:
        """
        Compute gradients of a single observable using the parameter‑shift rule.

        Parameters
        ----------
        observable: BaseOperator
            The target observable.
        parameter_sets: sequence of parameter sequences
            Points in parameter space at which to evaluate the gradient.
        shift: float, optional
            The shift angle used in the parameter‑shift rule.

        Returns
        -------
        List[List[complex]]
            Gradient vectors (one per parameter set) with respect to each circuit parameter.
        """
        grad_results: List[List[complex]] = []

        for params in parameter_sets:
            grads = []
            for idx, _ in enumerate(self.parameters):
                shifted_plus = list(params)
                shifted_minus = list(params)
                shifted_plus[idx] += shift
                shifted_minus[idx] -= shift

                exp_plus = self.evaluate([observable], [shifted_plus])[0][0]
                exp_minus = self.evaluate([observable], [shifted_minus])[0][0]

                grad = 0.5 * (exp_plus - exp_minus)
                grads.append(grad)
            grad_results.append(grads)

        return grad_results


__all__ = ["FastBaseEstimator"]
