import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Quantum estimator for parameterised circuits with support for shot noise and gradients.

    Parameters
    ----------
    circuit
        A parameterised :class:`qiskit.circuit.QuantumCircuit` that will be bound
        to the input parameters for each evaluation.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._backend = Aer.get_backend("statevector_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return a matrix of expectation values.

        The method uses the state‑vector simulator for exact computation.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circuit = self._bind(values)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Same as :meth:`evaluate` but runs the circuit on the Aer qasm simulator
        and returns noisy expectation values using the specified number of shots.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        backend = Aer.get_backend("qasm_simulator")
        for values in parameter_sets:
            circuit = self._bind(values)
            job = execute(
                circuit,
                backend,
                shots=shots,
                seed_simulator=seed,
                seed_transpiler=seed,
            )
            result = job.result()
            counts = result.get_counts(circuit)
            row = []
            for obs in observables:
                exp = self._counts_to_expectation(counts, obs)
                row.append(exp)
            results.append(row)
        return results

    def _counts_to_expectation(self, counts: dict[str, int], obs: BaseOperator) -> complex:
        """Compute expectation value from measurement counts for a Pauli observable."""
        if not isinstance(obs, Pauli):
            raise TypeError("Only Pauli observables are supported in the shot‑based method.")
        expectation = 0.0
        for bitstring, freq in counts.items():
            parity = 1
            for qubit, pauli in enumerate(obs.paulis[::-1]):
                if pauli == "I":
                    continue
                if bitstring[qubit] == "1":
                    parity *= -1
            expectation += parity * freq
        expectation /= sum(counts.values())
        return expectation

    def gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shift: float = np.pi / 2,
        shots: Optional[int] = None,
        seed: int | None = None,
    ) -> List[List[np.ndarray]]:
        """Return analytical gradients using the parameter‑shift rule.

        For each observable, the gradient is computed as

       .. math::
            \\frac{\\partial \\langle O \\rangle}{\\partial \\theta}
                = \\frac{1}{2}\\bigl(\\langle O \\rangle_{\\theta+\\delta}
                                 -\\langle O \\rangle_{\\theta-\\delta}\\bigr).

        Parameters
        ----------
        observables
            Iterable of Pauli observables.
        parameter_sets
            Sequence of parameter vectors.
        shift
            Shift angle for the parameter‑shift rule (default π/2).
        shots
            If provided, expectation values for the shifted circuits are
            evaluated using :meth:`evaluate_shots`; otherwise the exact
            state‑vector evaluator is used.
        seed
            Random seed for the shot simulator.
        """
        observables = list(observables)
        grads: List[List[np.ndarray]] = []
        for values in parameter_sets:
            grad_row: List[np.ndarray] = []
            for obs in observables:
                grad_vec = np.zeros(len(values), dtype=complex)
                for idx, theta in enumerate(values):
                    shifted_plus = list(values)
                    shifted_minus = list(values)
                    shifted_plus[idx] += shift
                    shifted_minus[idx] -= shift
                    if shots is None:
                        exp_plus = self.evaluate([obs], [shifted_plus])[0][0]
                        exp_minus = self.evaluate([obs], [shifted_minus])[0][0]
                    else:
                        exp_plus = self.evaluate_shots([obs], [shifted_plus], shots, seed)[0][0]
                        exp_minus = self.evaluate_shots([obs], [shifted_minus], shots, seed)[0][0]
                    grad_vec[idx] = 0.5 * (exp_plus - exp_minus)
                grad_row.append(grad_vec)
            grads.append(grad_row)
        return grads

__all__ = ["FastBaseEstimator"]
