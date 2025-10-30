"""Quantum expectation‑value estimator with noise, shots, and gradient support."""
from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel


class FastBaseEstimator:
    """Evaluate expectation values of a parametrized circuit.

    Features
    ----------
    * Noise‑model support via AerSimulator.
    * Automatic transpilation for the chosen backend.
    * Parameter‑shift gradient routine.
    * Vectorized evaluation over many parameter sets.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[AerSimulator] = None,
        noise_model: Optional[NoiseModel] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        self.original_circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        self.noise_model = noise_model
        self.shots = shots
        self.seed = seed

        self._transpiled = transpile(
            self.original_circuit,
            backend=self.backend,
            optimization_level=3,
        )

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.original_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of Qiskit Operator objects. If None a Pauli‑Z on the first qubit
            is used.
        parameter_sets:
            Iterable of parameter sequences. If None an empty list is returned.
        """
        if parameter_sets is None:
            return []

        if observables is None:
            from qiskit.quantum_info.operators import Pauli
            observables = [Pauli.from_label("Z")]

        results: List[List[complex]] = []

        for values in parameter_sets:
            circ = self._bind(values)
            circ = transpile(circ, backend=self.backend, optimization_level=3)

            job = self.backend.run(
                circ,
                noise_model=self.noise_model,
                shots=self.shots,
                seed_simulator=self.seed,
            )
            result = job.result()
            state = result.get_statevector(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

    def gradient(
        self,
        observable: Operator,
        parameter_index: int,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[float]:
        """Compute the gradient of an observable w.r.t. a single parameter using
        the parameter‑shift rule.

        Returns a list of gradients for each parameter set.
        """
        grads: List[float] = []
        for values in parameter_sets:
            base = list(values)
            plus = base.copy()
            minus = base.copy()
            plus[parameter_index] += shift
            minus[parameter_index] -= shift

            val_plus = self.evaluate([observable], [plus])[0][0]
            val_minus = self.evaluate([observable], [minus])[0][0]
            grads.append(0.5 * (val_plus - val_minus))
        return grads


__all__ = ["FastBaseEstimator"]
