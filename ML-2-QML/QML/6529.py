"""Enhanced FastBaseEstimator for quantum circuits with gradient, shots, noise, and caching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional, Dict

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import PauliSumOp


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit with advanced features."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Backend | None = None,
        noise_model: NoiseModel | None = None,
        shots: int | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.noise_model = noise_model
        self.shots = shots
        self._cache: Dict[Tuple[float,...], List[complex]] = {}

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _run(self, circuit: QuantumCircuit) -> List[complex]:
        """Execute a circuit and return expectation values for cached observables."""
        if isinstance(self.backend, Aer.StatevectorSimulator):
            state = Statevector.from_instruction(circuit)
            return [state.expectation_value(obs) for obs in self._observables]
        else:
            # Use qasm simulator with shots
            job = execute(
                circuit,
                backend=self.backend,
                shots=self.shots or 1024,
                noise_model=self.noise_model,
            )
            result = job.result()
            counts = result.get_counts()
            exp_vals = []
            for obs in self._observables:
                if isinstance(obs, BaseOperator):
                    exp_vals.append(result.get_expectation_value(obs, statevector=None))
                else:
                    raise TypeError("Unsupported observable type.")
            return exp_vals

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        self._observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            key = tuple(values)
            if key in self._cache:
                row = self._cache[key]
            else:
                circuit = self._bind(values)
                row = self._run(circuit)
                self._cache[key] = row
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return noisy estimates using shot‑based simulation."""
        self.shots = shots or self.shots or 1024
        if seed is not None:
            np.random.seed(seed)
        return self.evaluate(observables, parameter_sets)

    def evaluate_with_gradients(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[complex, List[complex]]]]:
        """
        Compute gradients of each observable w.r.t. each parameter using the
        parameter‑shift rule. Returns a list of rows, each containing tuples of
        (observable_value, [grad_sigma for each parameter]).
        """
        self._observables = list(observables)
        results: List[List[Tuple[complex, List[complex]]]] = []
        shift = np.pi / 2
        for values in parameter_sets:
            base_circuit = self._bind(values)
            base_vals = self._run(base_circuit)
            grads: List[List[complex]] = [[] for _ in self._observables]
            for i, param in enumerate(self._parameters):
                # Shift positive
                pos_vals = list(values)
                pos_vals[i] += shift
                pos_circuit = self._bind(pos_vals)
                pos_exp = self._run(pos_circuit)
                # Shift negative
                neg_vals = list(values)
                neg_vals[i] -= shift
                neg_circuit = self._bind(neg_vals)
                neg_exp = self._run(neg_circuit)
                # Compute gradient
                for j, obs in enumerate(self._observables):
                    grad = (pos_exp[j] - neg_exp[j]) / (2 * np.sin(shift))
                    grads[j].append(grad)
            row = [(base_vals[j], grads[j]) for j in range(len(self._observables))]
            results.append(row)
        return results

    def cache_parameters(self, parameter_sets: Sequence[Sequence[float]]) -> None:
        """Cache parameter sets for quick repeated evaluation."""
        for values in parameter_sets:
            key = tuple(values)
            if key not in self._cache:
                circuit = self._bind(values)
                self._cache[key] = self._run(circuit)


__all__ = ["FastBaseEstimator"]
